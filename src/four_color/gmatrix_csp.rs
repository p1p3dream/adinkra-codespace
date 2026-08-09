#![allow(dead_code)]
//! v2 slot-level CSP engine for the full CLS G-matrix enumeration
//! (G^2 = A over {-1,0,1}; see gmatrix_full.rs for the problem and the K =
//! Q(sqrt(-3)) commutant reduction). Where gmatrix_full assigns the 36 entries
//! of g one at a time and prunes with ENTRYWISE-independent product boxes,
//! this engine assigns the nine 2x2 slots atomically and prunes with
//! CORRELATED constraints:
//!
//!   1. Slot domains are bitsets over the <=825 integral slot values (the
//!      images of the {-1,0,1} intertwiners), never partial entry guesses.
//!   2. Each block equation sum_k slot(bi,k) * slot(k,bj) = lambda*delta_ij*I2
//!      is treated as one 2x2 matrix equation. Per-term achievable boxes are
//!      min/max over the ACTUAL slot-pair product set (precomputed per entry
//!      as widened f64), not over the cross product of entry alphabets.
//!   3. When all but one term of a block equation is assigned, the remaining
//!      term must hit the exact residual R: slot values with no partner
//!      achieving X*Y = R are removed (arc consistency). Stage 1 is a
//!      precomputed product-to-support map lookup per slot pair (one hash
//!      hit plus two bitset ANDs); stage 2 refines to full pairwise arc
//!      consistency with an f64-prefiltered scan when the surviving
//!      candidate rectangle is small.
//!   4. The trace(g) = 0 condition is a slot-trace bracket.
//!   5. Assignment order is dynamic: smallest surviving domain (MRV).
//!
//! Soundness: f64 boxes are widened outward (never discard a true product),
//! residual filtering removes a value only when NO partner in the current
//! domain achieves the exact residual, and every leaf is re-verified exactly
//! (g^2 = lambda I over K, then integer map-back and G^2 = A). Domains start
//! at the complete integral slot sets, so no integer solution is ever pruned.
//!
//! When all slot value lists are identical across slots (true when the three
//! CLS blocks are equal, detected at build time), one shared product table
//! serves every slot pair; otherwise per-pair tables are built.
//!
//! Checksum compatibility: identical splitmix64-sum of hash_intmat over
//! verified integer G matrices as gmatrix_full, so v1/v2 artifacts are
//! directly comparable (m=2 target: count 15000, checksum 94e85bc1c8e786fd).

use super::gmatrix_full::{
    Alphabets, Coords, K, Side, build_alphabets, build_coords, cls_a_blocks, hash_intmat,
    int_of_g, slot_int_of_g, slot_solutions, block_diagonal_via_coords,
};
use super::{gmatrix, imm, IntMat};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

// ===========================================================================
// Bitsets over slot-value indices (825 values max => 13 u64 words)
// ===========================================================================

const SW: usize = 13;
type Bits = [u64; SW];

fn b_test(b: &Bits, i: usize) -> bool {
    (b[i / 64] >> (i % 64)) & 1 == 1
}
fn b_set(b: &mut Bits, i: usize) {
    b[i / 64] |= 1u64 << (i % 64);
}
fn b_empty(b: &Bits) -> bool {
    b.iter().all(|&w| w == 0)
}
fn b_popc(b: &Bits) -> u32 {
    b.iter().map(|w| w.count_ones()).sum()
}
fn b_singleton(b: &Bits) -> Option<usize> {
    let mut found = None;
    for (wi, &w) in b.iter().enumerate() {
        if w == 0 {
            continue;
        }
        if found.is_some() || w & (w - 1) != 0 {
            return None;
        }
        found = Some(wi * 64 + w.trailing_zeros() as usize);
    }
    found
}
fn b_full(n: usize) -> Bits {
    let mut b = [0u64; SW];
    for i in 0..n {
        b_set(&mut b, i);
    }
    b
}
fn b_each(b: &Bits, mut f: impl FnMut(usize)) {
    for (wi, &w0) in b.iter().enumerate() {
        let mut w = w0;
        while w != 0 {
            let tz = w.trailing_zeros() as usize;
            f(wi * 64 + tz);
            w &= w - 1;
        }
    }
}

// ===========================================================================
// Exact 2x2 K-matrix helpers (residual solves and leaf checks)
// ===========================================================================

type K2 = [[K; 2]; 2];

fn mul22(a: &K2, b: &K2) -> K2 {
    [
        [
            a[0][0].mul(b[0][0]).add(a[0][1].mul(b[1][0])),
            a[0][0].mul(b[0][1]).add(a[0][1].mul(b[1][1])),
        ],
        [
            a[1][0].mul(b[0][0]).add(a[1][1].mul(b[1][0])),
            a[1][0].mul(b[0][1]).add(a[1][1].mul(b[1][1])),
        ],
    ]
}

fn inv22(a: &K2) -> Option<K2> {
    let det = a[0][0].mul(a[1][1]).sub(a[0][1].mul(a[1][0]));
    if det.is_zero() {
        return None;
    }
    let d = det.inv();
    Some([
        [a[1][1].mul(d), a[0][1].neg().mul(d)],
        [a[1][0].neg().mul(d), a[0][0].mul(d)],
    ])
}

/// f64 approximation of (re, im); only ever used inside outward-widened boxes.
fn kf64(x: K) -> (f64, f64) {
    let ((rn, rd), (inn, ind)) = x.parts();
    (rn as f64 / rd as f64, inn as f64 / ind as f64)
}

/// Outward padding for box bounds: covers the rational->f64 conversion error
/// (<= 0.5 ulp) with three orders of magnitude to spare.
fn pad(x: f64) -> f64 {
    x.abs() * 1e-12 + 1e-300
}

// ===========================================================================
// Shared precomputation: slot lists, product tables, support maps, traces
// ===========================================================================

/// Per-entry product approximations for one ordered pair of slot value lists.
/// re/im[x * ny + y][e] is the f64 value of entry e (u = e/2, w = e%2) of
/// slot_a[x] * slot_b[y]. Used only inside widened min/max boxes.
/// The flo/fhi fields are the min/max over ALL pairs (the full-domain box);
/// always a sound superset of any partial-domain achievable set.
struct PairTable {
    ny: usize,
    re: Vec<[f64; 4]>,
    im: Vec<[f64; 4]>,
    flo_re: [f64; 4],
    fhi_re: [f64; 4],
    flo_im: [f64; 4],
    fhi_im: [f64; 4],
}

pub struct Shared {
    m: usize,
    n: usize,
    coords: Coords,
    slots: Vec<Vec<K2>>,
    /// tables[l * (m*m) + r], None when the pair never appears in a product.
    /// When all slot lists are identical, only table index 0 is built and
    /// `shared_table` is true (every lookup uses index 0).
    tables: Vec<Option<PairTable>>,
    /// Product-to-support lookup maps, same indexing as `tables`: for each
    /// exact product P, the set of left values x with some partner y giving
    /// x*y = P, and symmetrically the right values. Diagonal pairs hold
    /// squares only (x, x). Measured at 352098 distinct products across all
    /// 27 pairs for m=3, about 140MB.
    pmaps: Vec<Option<HashMap<K2, (Bits, Bits)>>>,
    shared_table: bool,
    /// Per diagonal slot: per-value trace (re, im) as f64.
    tr_re: Vec<Vec<f64>>,
    tr_im: Vec<Vec<f64>>,
}

fn build_pair(a: &[K2], b: &[K2], square: bool) -> (PairTable, HashMap<K2, (Bits, Bits)>) {
    let ny = b.len();
    let mut re = Vec::with_capacity(a.len() * ny);
    let mut im = Vec::with_capacity(a.len() * ny);
    let mut flo_re = [f64::INFINITY; 4];
    let mut fhi_re = [f64::NEG_INFINITY; 4];
    let mut flo_im = [f64::INFINITY; 4];
    let mut fhi_im = [f64::NEG_INFINITY; 4];
    let mut pmap: HashMap<K2, (Bits, Bits)> = HashMap::new();
    for (xi, x) in a.iter().enumerate() {
        for (yi, y) in b.iter().enumerate() {
            let p = mul22(x, y);
            if !square || xi == yi {
                // Off-diagonal pairs record every product; square pairs
                // record only the diagonal (x, x) squares actually achievable
                // by a square term.
                let e = pmap.entry(p).or_insert_with(|| ([0u64; SW], [0u64; SW]));
                b_set(&mut e.0, xi);
                b_set(&mut e.1, yi);
            }
            let mut pr = [0f64; 4];
            let mut pi = [0f64; 4];
            for u in 0..2 {
                for w in 0..2 {
                    let (r, i) = kf64(p[u][w]);
                    pr[2 * u + w] = r;
                    pi[2 * u + w] = i;
                }
            }
            for e in 0..4 {
                flo_re[e] = flo_re[e].min(pr[e]);
                fhi_re[e] = fhi_re[e].max(pr[e]);
                flo_im[e] = flo_im[e].min(pi[e]);
                fhi_im[e] = fhi_im[e].max(pi[e]);
            }
            re.push(pr);
            im.push(pi);
        }
    }
    for e in 0..4 {
        flo_re[e] -= pad(flo_re[e]);
        fhi_re[e] += pad(fhi_re[e]);
        flo_im[e] -= pad(flo_im[e]);
        fhi_im[e] += pad(fhi_im[e]);
    }
    (
        PairTable {
            ny,
            re,
            im,
            flo_re,
            fhi_re,
            flo_im,
            fhi_im,
        },
        pmap,
    )
}

fn build_shared(coords: Coords, alph: &Alphabets) -> Shared {
    let m = coords.m;
    let mm = m * m;
    let slots: Vec<Vec<K2>> = alph.slots.clone();
    let nvals = slots[0].len();
    assert!(
        slots.iter().all(|s| s.len() == nvals),
        "slot value counts must agree across slots"
    );
    let shared_table = slots.iter().all(|s| *s == slots[0]);
    let mut tables: Vec<Option<PairTable>> = (0..mm * mm).map(|_| None).collect();
    let mut pmaps: Vec<Option<HashMap<K2, (Bits, Bits)>>> = (0..mm * mm).map(|_| None).collect();
    if shared_table {
        let (t, pm) = build_pair(&slots[0], &slots[0], m == 1);
        tables[0] = Some(t);
        pmaps[0] = Some(pm);
    } else {
        for bi in 0..m {
            for bk in 0..m {
                for bj in 0..m {
                    let (l, r) = (bi * m + bk, bk * m + bj);
                    let idx = l * mm + r;
                    if tables[idx].is_none() {
                        let (t, pm) = build_pair(&slots[l], &slots[r], l == r);
                        tables[idx] = Some(t);
                        pmaps[idx] = Some(pm);
                    }
                }
            }
        }
    }
    let mut tr_re = Vec::with_capacity(m);
    let mut tr_im = Vec::with_capacity(m);
    for bi in 0..m {
        let s = &slots[bi * m + bi];
        tr_re.push(
            s.iter()
                .map(|x| kf64(x[0][0].add(x[1][1])).0)
                .collect(),
        );
        tr_im.push(
            s.iter()
                .map(|x| kf64(x[0][0].add(x[1][1])).1)
                .collect(),
        );
    }
    Shared {
        m,
        n: 2 * m,
        coords,
        slots,
        tables,
        pmaps,
        shared_table,
        tr_re,
        tr_im,
    }
}

impl Shared {
    fn table(&self, l: usize, r: usize) -> &PairTable {
        let mm = self.m * self.m;
        let idx = if self.shared_table { 0 } else { l * mm + r };
        self.tables[idx].as_ref().expect("pair table must exist")
    }

    fn pmap(&self, l: usize, r: usize) -> &HashMap<K2, (Bits, Bits)> {
        let mm = self.m * self.m;
        let idx = if self.shared_table { 0 } else { l * mm + r };
        self.pmaps[idx].as_ref().expect("pair map must exist")
    }
}

// ===========================================================================
// Structural classification at solution leaves
// ===========================================================================

/// Exact integer determinants (i64; block entries are trits, so any 4x4
/// minor has |det| <= 24 and i64 is vastly more than enough).
fn det2(a: &[[i64; 4]; 4], r: [usize; 2], c: [usize; 2]) -> i64 {
    a[r[0]][c[0]] * a[r[1]][c[1]] - a[r[0]][c[1]] * a[r[1]][c[0]]
}

fn det3(a: &[[i64; 4]; 4], r: [usize; 3], c: [usize; 3]) -> i64 {
    a[r[0]][c[0]] * det2(a, [r[1], r[2]], [c[1], c[2]])
        - a[r[0]][c[1]] * det2(a, [r[1], r[2]], [c[0], c[2]])
        + a[r[0]][c[2]] * det2(a, [r[1], r[2]], [c[0], c[1]])
}

fn det4(a: &[[i64; 4]; 4]) -> i64 {
    a[0][0] * det3(a, [1, 2, 3], [1, 2, 3])
        - a[0][1] * det3(a, [1, 2, 3], [0, 2, 3])
        + a[0][2] * det3(a, [1, 2, 3], [0, 1, 3])
        - a[0][3] * det3(a, [1, 2, 3], [0, 1, 2])
}

/// Exact rank of a 4x4 trit block via minor checks (no division at all).
fn rank4(b: &[[i64; 4]; 4]) -> u8 {
    if det4(b) != 0 {
        return 4;
    }
    for i in 0..4 {
        for j in 0..4 {
            let r: Vec<usize> = (0..4).filter(|&x| x != i).collect();
            let c: Vec<usize> = (0..4).filter(|&x| x != j).collect();
            if det3(b, [r[0], r[1], r[2]], [c[0], c[1], c[2]]) != 0 {
                return 3;
            }
        }
    }
    for r1 in 0..4 {
        for r2 in (r1 + 1)..4 {
            for c1 in 0..4 {
                for c2 in (c1 + 1)..4 {
                    if det2(b, [r1, r2], [c1, c2]) != 0 {
                        return 2;
                    }
                }
            }
        }
    }
    if b.iter().flatten().any(|&x| x != 0) {
        return 1;
    }
    0
}

/// Structural class of a solution: total nonzeros, the 4x4-block support
/// pattern (bit bi*m+bj), and the exact rank of every 4x4 block. Diagonal
/// versus cross-block structure is read off the support mask.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ClassKey {
    pub nnz: u32,
    pub support: u16,
    pub ranks: [u8; 9],
}

pub struct ClassRec {
    pub count: u64,
    pub checksum: u64,
    pub rep_hash: u64,
    pub rep: IntMat,
}

fn class_key(gi: &IntMat, m: usize) -> ClassKey {
    let mut nnz = 0u32;
    let mut support = 0u16;
    let mut ranks = [0xFFu8; 9];
    for bi in 0..m {
        for bj in 0..m {
            let mut b = [[0i64; 4]; 4];
            for u in 0..4 {
                for w in 0..4 {
                    b[u][w] = gi[4 * bi + u][4 * bj + w] as i64;
                }
            }
            let r = rank4(&b);
            ranks[bi * m + bj] = r;
            if r > 0 {
                support |= 1u16 << (bi * m + bj);
            }
            nnz += b.iter().flatten().filter(|&&x| x != 0).count() as u32;
        }
    }
    ClassKey {
        nnz,
        support,
        ranks,
    }
}

/// Merge worker histograms commutatively; representative = min hash, so the
/// choice is deterministic across thread counts and schedulings.
fn merge_hist(into: &mut HashMap<ClassKey, ClassRec>, from: HashMap<ClassKey, ClassRec>) {
    for (k, rec) in from {
        match into.entry(k) {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                let r = e.get_mut();
                r.count += rec.count;
                r.checksum = r.checksum.wrapping_add(rec.checksum);
                if rec.rep_hash < r.rep_hash {
                    r.rep_hash = rec.rep_hash;
                    r.rep = rec.rep;
                }
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(rec);
            }
        }
    }
}

// ===========================================================================
// Search state: domains, versions, cached term boxes
// ===========================================================================

/// A cached achievable box for one term. Valid for any current domain pair
/// contained in (snap_l, snap_r): boxes are min/max, so a superset snapshot
/// yields a sound (possibly loose) box. Subset validity survives backtracks
/// automatically, no version stamps needed.
#[derive(Clone, Copy)]
struct TermBox {
    have: bool,
    snap_l: Bits,
    snap_r: Bits,
    lo_re: [f64; 4],
    hi_re: [f64; 4],
    lo_im: [f64; 4],
    hi_im: [f64; 4],
}

impl TermBox {
    fn empty() -> TermBox {
        TermBox {
            have: false,
            snap_l: [0; SW],
            snap_r: [0; SW],
            lo_re: [0.0; 4],
            hi_re: [0.0; 4],
            lo_im: [0.0; 4],
            hi_im: [0.0; 4],
        }
    }

    fn valid_for(&self, dl: &Bits, dr: &Bits) -> bool {
        self.have && b_subset(dl, &self.snap_l) && b_subset(dr, &self.snap_r)
    }
}

fn b_subset(a: &Bits, b: &Bits) -> bool {
    a.iter().zip(b.iter()).all(|(x, y)| x & !y == 0)
}

/// Widened f64 bracket test for one block equation: assigned exact sum plus
/// per-term achievable boxes must contain the target in every scalar entry.
/// Sound for pruning because every box is outward-widened and covers a
/// superset of the term's achievable products.
fn bracket_ok(sum: &[K; 4], tbs: &[TermBox], bi: usize, bj: usize, lam: K) -> bool {
    for u in 0..2 {
        for w in 0..2 {
            let e = 2 * u + w;
            let target = if bi == bj && u == w { lam } else { K::zero() };
            let (tr, ti) = kf64(target);
            let (sr, si) = kf64(sum[e]);
            let (mut lo_re, mut hi_re) = (sr - pad(sr), sr + pad(sr));
            let (mut lo_im, mut hi_im) = (si - pad(si), si + pad(si));
            for tb in tbs {
                lo_re += tb.lo_re[e];
                hi_re += tb.hi_re[e];
                lo_im += tb.lo_im[e];
                hi_im += tb.hi_im[e];
            }
            if lo_re > tr || hi_re < tr || lo_im > ti || hi_im < ti {
                return false;
            }
        }
    }
    true
}

/// Tighten-on-the-cheap threshold: when a check passes on the wide box only,
/// compute the exact tight box if the scan costs at most this many table
/// pairs. Above it, accept the sound wide pass instead of paying an
/// 825x825-scale rescan at the top of the tree. Runtime-tunable via
/// ADINKRA_CSP_TIGHT: a huge value forces always-tighten, reproducing the
/// pruning verdicts of the always-fresh policy for A/B validation.
fn tight_max_pairs() -> usize {
    use std::sync::OnceLock;
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("ADINKRA_CSP_TIGHT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16384)
    })
}

/// Candidate-rectangle cap for the stage-2 pairwise arc-consistency refine
/// in the residual filter. Rectangles at or below this size get the full
/// partner-in-current-domain scan; larger ones keep the stage-1 global
/// support sets. ADINKRA_CSP_REFINE overrides.
fn refine_max_pairs() -> usize {
    use std::sync::OnceLock;
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("ADINKRA_CSP_REFINE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(65536)
    })
}

// ===========================================================================
// Live observability: shared counters, per-worker slots, heartbeat thread.
// Every increment is a relaxed atomic (unmeasurable at m=3 rates, ~1.5ms per
// node). Nothing here influences search decisions; node counts are untouched.
// ===========================================================================

/// Stride-100 m=3 reference node counts (results/cls-gmatrix-v2-m3-stride100-benchmark.md).
/// Display-only: pct-of-expected on reruns of known prefixes.
fn ref_nodes_m3(item: u32) -> Option<u64> {
    // Authoritative node counts from the validated stride-100 reference
    // shards in results/cls_g_csp_shards_L_3blocks_s100 (ladder-verified).
    Some(match item {
        0 => 14_010_043,
        100 => 783_979,
        200 => 5_828_587,
        300 => 755_179,
        400 => 2_629_003,
        500 => 5_067_179,
        600 => 5_828_587,
        700 => 719_467,
        800 => 811_627,
        _ => return None,
    })
}

/// Per-worker in-flight state, one slot per search thread. Written by the
/// worker, read only by the heartbeat thread.
struct WorkerSlot {
    /// Current item id, or u64::MAX when idle.
    item: AtomicU64,
    /// Nodes searched on the current item.
    nodes: AtomicU64,
    depth: AtomicU64,
    max_depth: AtomicU64,
    /// Millis on the run clock when the current item started.
    started_ms: AtomicU64,
    /// Seqlock guarding the whole slot: odd while item_start/item_done are
    /// mutating, even when stable. A heartbeat tick that sees an odd or
    /// changing sequence skips the worker rather than reporting a torn mix
    /// of one item's id and another's counters.
    seq: AtomicU64,
}

/// Pod identity for per-pod status/tmp naming.
fn pod_name() -> String {
    std::env::var("ADINKRA_CSP_POD")
        .or_else(|_| std::env::var("HOSTNAME"))
        .unwrap_or_else(|_| "local".to_string())
}

/// Shared live statistics for one solve or shard run.
struct LiveStats {
    t0: Instant,
    pod: String,
    /// Block count of the run; the stride-100 reference table is m=3 only,
    /// so ref_pct display is gated on this.
    m: usize,
    nodes: AtomicU64,
    eq_checks: AtomicU64,
    stage1: AtomicU64,
    stage1_miss: AtomicU64,
    stage2: AtomicU64,
    props: AtomicU64,
    tight: AtomicU64,
    leaves: AtomicU64,
    solutions: AtomicU64,
    items_done: AtomicU64,
    items_total: u64,
    /// Nodes by recursion depth (depths past 31 fold into the last bucket).
    depth_hist: [AtomicU64; 32],
    workers: Vec<WorkerSlot>,
}

impl LiveStats {
    fn new(items_total: u64, nworkers: usize, m: usize) -> LiveStats {
        let pod = pod_name();
        LiveStats {
            t0: Instant::now(),
            pod,
            m,
            nodes: AtomicU64::new(0),
            eq_checks: AtomicU64::new(0),
            stage1: AtomicU64::new(0),
            stage1_miss: AtomicU64::new(0),
            stage2: AtomicU64::new(0),
            props: AtomicU64::new(0),
            tight: AtomicU64::new(0),
            leaves: AtomicU64::new(0),
            solutions: AtomicU64::new(0),
            items_done: AtomicU64::new(0),
            items_total,
            depth_hist: [const { AtomicU64::new(0) }; 32],
            workers: (0..nworkers.max(1))
                .map(|_| WorkerSlot {
                    item: AtomicU64::new(u64::MAX),
                    nodes: AtomicU64::new(0),
                    depth: AtomicU64::new(0),
                    max_depth: AtomicU64::new(0),
                    started_ms: AtomicU64::new(0),
                    seq: AtomicU64::new(0),
                })
                .collect(),
        }
    }

    fn item_start(&self, wid: usize, item: u32) {
        let w = &self.workers[wid];
        // Seqlock write: bracket the mutation with odd/even sequence values.
        // The item id is stored last, inside the bracket.
        w.seq.fetch_add(1, Ordering::AcqRel);
        w.nodes.store(0, Ordering::Relaxed);
        w.depth.store(0, Ordering::Relaxed);
        w.max_depth.store(0, Ordering::Relaxed);
        w.started_ms
            .store(self.t0.elapsed().as_millis() as u64, Ordering::Relaxed);
        w.item.store(item as u64, Ordering::Relaxed);
        w.seq.fetch_add(1, Ordering::AcqRel);
    }

    fn item_done(&self, wid: usize) {
        let w = &self.workers[wid];
        w.seq.fetch_add(1, Ordering::AcqRel);
        w.item.store(u64::MAX, Ordering::Relaxed);
        // items_done stays inside the bracket so a tick that observes the
        // idle worker never lags the global completion counter.
        self.items_done.fetch_add(1, Ordering::Relaxed);
        w.seq.fetch_add(1, Ordering::AcqRel);
    }

    /// Mark the worker idle WITHOUT counting completion: used when an
    /// item's durable install fails and the item is left for resume, so the
    /// final record never shows a phantom in-flight worker on a failed item.
    fn item_abandon(&self, wid: usize) {
        let w = &self.workers[wid];
        w.seq.fetch_add(1, Ordering::AcqRel);
        w.item.store(u64::MAX, Ordering::Relaxed);
        w.seq.fetch_add(1, Ordering::AcqRel);
    }
}

/// Heartbeat interval in seconds; ADINKRA_CSP_HEARTBEAT overrides, 0 disables.
fn heartbeat_secs() -> u64 {
    use std::sync::OnceLock;
    static V: OnceLock<u64> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("ADINKRA_CSP_HEARTBEAT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(15)
    })
}

/// Resident memory in MB. Linux reads /proc; other platforms report 0
/// (the fleet runs on Linux, where this matters).
fn rss_mb() -> f64 {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("VmRSS"))
                    .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse::<f64>().ok()))
            })
            .map(|kb| kb / 1024.0)
            .unwrap_or(0.0)
    }
    #[cfg(not(target_os = "linux"))]
    {
        0.0
    }
}

/// One heartbeat's shared state: each emission is one human line to stderr
/// plus, when status_path is set, a durable JSON record (one jsonl append
/// plus an atomic snapshot, one file per process on a shared volume). Pure
/// reader of LiveStats; never touches search. The Mutex serializes the
/// sleeping thread's ticks with the caller's final synchronous emit at run
/// end.
struct Heartbeat {
    live: Arc<LiveStats>,
    status_path: Option<String>,
    /// Tick deltas: nodes, per-depth counts, and the previous emission time.
    state: std::sync::Mutex<(u64, [u64; 32], Instant)>,
}

impl Heartbeat {
    fn emit(&self) {
        use std::io::Write as _;
        let live = &self.live;
        let mut g = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let (last_nodes, last_hist, last_emit) = &mut *g;
        let up = live.t0.elapsed().as_secs_f64();
        let now = Instant::now();
        // The per-tick rate divides by the REAL elapsed time since the last
        // emission: the final synchronous emit rarely lands exactly one
        // configured interval after the previous tick.
        let tick_secs = now.duration_since(*last_emit).as_secs_f64().max(1e-9);
        *last_emit = now;
        let nd = live.nodes.load(Ordering::Relaxed);
        let tick = nd - *last_nodes;
        *last_nodes = nd;
        let rate_avg = nd as f64 / up.max(1e-9);
        let rate_tick = tick as f64 / tick_secs;
        let sols = live.solutions.load(Ordering::Relaxed);
        let s1 = live.stage1.load(Ordering::Relaxed);
        let s1m = live.stage1_miss.load(Ordering::Relaxed);
        // stage1 counts every lookup attempt (misses included), so the
        // hit rate is (attempts - misses) / attempts.
        let s1_hit = if s1 > 0 {
            100.0 * s1.saturating_sub(s1m) as f64 / s1 as f64
        } else {
            100.0
        };
        let mut working = 0usize;
        let mut wjson = Vec::new();
        let now_ms = live.t0.elapsed().as_millis() as u64;
        for (wid, w) in live.workers.iter().enumerate() {
            // Seqlock read: skip any worker caught mid-transition, and
            // re-verify the sequence after the reads so a torn snapshot
            // is never reported.
            let s0 = w.seq.load(Ordering::Acquire);
            if s0 % 2 == 1 {
                continue;
            }
            let item = w.item.load(Ordering::Relaxed);
            let wn = w.nodes.load(Ordering::Relaxed);
            let wdepth = w.depth.load(Ordering::Relaxed);
            let wmaxd = w.max_depth.load(Ordering::Relaxed);
            let wstart = w.started_ms.load(Ordering::Relaxed);
            // Acquire fence before the confirming sequence load: on
            // weak-memory targets this keeps the second seq load from
            // being performed ahead of the field loads above.
            std::sync::atomic::fence(Ordering::Acquire);
            if s0 != w.seq.load(Ordering::Acquire) {
                continue;
            }
            if item == u64::MAX {
                continue;
            }
            working += 1;
            let wsecs = now_ms.saturating_sub(wstart) / 1000;
            // The reference table holds m=3 stride-100 node counts;
            // attach ref_pct only for actual m=3 runs.
            let ref_pct = if live.m == 3 {
                ref_nodes_m3(item as u32).map(|r| 100.0 * wn as f64 / r as f64)
            } else {
                None
            };
            wjson.push(serde_json::json!({
                "wid": wid,
                "item": item,
                "nodes": wn,
                "secs_on_item": wsecs,
                "depth": wdepth,
                "max_depth": wmaxd,
                "ref_pct_of_expected": ref_pct,
            }));
        }
        // Read the global completion counter only after the worker
        // snapshots: item_done increments it inside the same seqlock
        // bracket that marks the worker idle, so a tick that observes an
        // idle worker never reports a stale count.
        let items_done = live.items_done.load(Ordering::Relaxed);
        let hist: Vec<u64> = live
            .depth_hist
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let v = c.load(Ordering::Relaxed);
                let d = v - last_hist[i];
                last_hist[i] = v;
                d
            })
            .collect();
        let eta_s = if items_done > 0 && items_done < live.items_total {
            Some((live.items_total - items_done) as f64 * up / items_done as f64)
        } else {
            None
        };
        eprintln!(
            "[hb {:>7.0}s] nodes={} (+{}/tick, {:.0}/s avg) items {}/{} done, {} working, sols={} stage1={:.1}%hit tight={} rss={:.0}MB eta={}",
            up,
            nd,
            tick,
            rate_avg,
            items_done,
            live.items_total,
            working,
            sols,
            s1_hit,
            live.tight.load(Ordering::Relaxed),
            rss_mb(),
            eta_s.map(|e| format!("{:.1}h", e / 3600.0)).unwrap_or_else(|| "?".to_string()),
        );
        if let Some(p) = &self.status_path {
            let line = serde_json::json!({
                "ts": up,
                "pod": live.pod,
                "nodes": nd,
                "rate_tick": rate_tick,
                "rate_avg": rate_avg,
                "solutions": sols,
                "items_done": items_done,
                "items_total": live.items_total,
                "eq_checks": live.eq_checks.load(Ordering::Relaxed),
                "stage1": s1,
                "stage1_miss": s1m,
                "stage2": live.stage2.load(Ordering::Relaxed),
                "props": live.props.load(Ordering::Relaxed),
                "tight": live.tight.load(Ordering::Relaxed),
                "leaves": live.leaves.load(Ordering::Relaxed),
                "depth_hist_tick": hist,
                "workers": wjson,
                "rss_mb": rss_mb(),
                "eta_s": eta_s,
            });
            // History log, rotated at 64MB (single .1 generation) so a
            // 1s-heartbeat multi-week run cannot fill the volume.
            if let Ok(md) = std::fs::metadata(p) {
                if md.len() > 64 * 1024 * 1024 {
                    let _ = std::fs::rename(p, format!("{p}.1"));
                }
            }
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(p) {
                let _ = writeln!(f, "{line}");
            }
            // Atomic latest-snapshot alongside the log: readers get a
            // guaranteed-complete record without scanning the history.
            let snap = p
                .strip_suffix(".jsonl")
                .map(|s| format!("{s}.json"))
                .unwrap_or_else(|| format!("{p}.snapshot"));
            let snap_tmp = format!("{snap}.tmp");
            if std::fs::write(&snap_tmp, format!("{line}\n")).is_ok() {
                let _ = std::fs::rename(&snap_tmp, &snap);
            }
        }
    }
}

/// Spawn the heartbeat thread: every interval emits one record. The caller
/// must call the returned handle's emit() after setting done: the sleeping
/// thread is killed at process exit without noticing done promptly, so
/// without a final synchronous emit a run shorter than one interval would
/// leave no status record at all.
fn spawn_heartbeat(
    live: Arc<LiveStats>,
    status_path: Option<String>,
    done: Arc<AtomicBool>,
) -> Option<Arc<Heartbeat>> {
    let secs = heartbeat_secs();
    if secs == 0 {
        return None;
    }
    let hb = Arc::new(Heartbeat {
        live,
        status_path,
        state: std::sync::Mutex::new((0, [0u64; 32], Instant::now())),
    });
    let hb_thread = Arc::clone(&hb);
    std::thread::spawn(move || loop {
        std::thread::sleep(std::time::Duration::from_secs(secs));
        if done.load(Ordering::Relaxed) {
            break;
        }
        hb_thread.emit();
    });
    Some(hb)
}

struct Search {
    dom: Vec<Bits>,
    /// Tight boxes: recomputed over small domains, invalidated by backtracks.
    tcache: Vec<TermBox>, // (bi*m+bj)*m+bk
    /// Wide boxes: seeded once per worker over its base domains (plus the
    /// precomputed full-domain box), valid for the worker's entire subtree.
    wcache: Vec<TermBox>,
    nodes: u64,
    count: u64,
    checksum: u64,
    tight_computes: u64,
    /// AC-3 style propagation queue: slots whose domains shrank since their
    /// incident equations were last checked. Filtering operators are
    /// monotone, so chaotic iteration reaches the same greatest fixpoint as
    /// full-scan propagation, at a fraction of the cost.
    dirty: Vec<bool>,
    queue: Vec<usize>,
    hist: HashMap<ClassKey, ClassRec>,
    samples: Vec<IntMat>,
    /// Live observability handle; None in tests and standalone searches.
    /// Read/increment only; never consulted by any search decision.
    live: Option<Arc<LiveStats>>,
    /// Worker slot index into live.workers; 0 when live is None.
    wid: usize,
    /// Current recursion depth (root call = 0). Display-only.
    depth: u32,
}

impl Search {
    fn new(sh: &Shared) -> Search {
        let mm = sh.m * sh.m;
        let nvals = sh.slots[0].len();
        Search {
            dom: (0..mm).map(|_| b_full(nvals)).collect(),
            tcache: (0..mm * sh.m).map(|_| TermBox::empty()).collect(),
            wcache: (0..mm * sh.m).map(|_| TermBox::empty()).collect(),
            nodes: 0,
            count: 0,
            checksum: 0,
            tight_computes: 0,
            dirty: vec![false; mm],
            queue: Vec::new(),
            hist: HashMap::new(),
            samples: Vec::new(),
            live: None,
            wid: 0,
            depth: 0,
        }
    }

    fn mark_dirty(&mut self, p: usize) {
        if !self.dirty[p] {
            self.dirty[p] = true;
            self.queue.push(p);
        }
    }

    /// Exact achievable box over the CURRENT domains (outward-widened).
    /// Stored in the tight cache with the current domains as snapshot.
    fn term_box_tight(&mut self, sh: &Shared, l: usize, r: usize, ci: usize) -> TermBox {
        self.tight_computes += 1;
        if let Some(live) = &self.live {
            live.tight.fetch_add(1, Ordering::Relaxed);
        }
        let t = sh.table(l, r);
        let mut lo_re = [f64::INFINITY; 4];
        let mut hi_re = [f64::NEG_INFINITY; 4];
        let mut lo_im = [f64::INFINITY; 4];
        let mut hi_im = [f64::NEG_INFINITY; 4];
        if l == r {
            // Square term: only diagonal (x, x) products are achievable.
            b_each(&self.dom[l], |x| {
                let i = x * t.ny + x;
                for e in 0..4 {
                    lo_re[e] = lo_re[e].min(t.re[i][e]);
                    hi_re[e] = hi_re[e].max(t.re[i][e]);
                    lo_im[e] = lo_im[e].min(t.im[i][e]);
                    hi_im[e] = hi_im[e].max(t.im[i][e]);
                }
            });
        } else {
            b_each(&self.dom[l], |x| {
                let base = x * t.ny;
                b_each(&self.dom[r], |y| {
                    let i = base + y;
                    for e in 0..4 {
                        lo_re[e] = lo_re[e].min(t.re[i][e]);
                        hi_re[e] = hi_re[e].max(t.re[i][e]);
                        lo_im[e] = lo_im[e].min(t.im[i][e]);
                        hi_im[e] = hi_im[e].max(t.im[i][e]);
                    }
                });
            });
        }
        for e in 0..4 {
            lo_re[e] -= pad(lo_re[e]);
            hi_re[e] += pad(hi_re[e]);
            lo_im[e] -= pad(lo_im[e]);
            hi_im[e] += pad(hi_im[e]);
        }
        let tb = TermBox {
            have: true,
            snap_l: self.dom[l],
            snap_r: self.dom[r],
            lo_re,
            hi_re,
            lo_im,
            hi_im,
        };
        self.tcache[ci] = tb;
        tb
    }

    /// Seed the wide box over the worker's base domains: the precomputed
    /// full-domain box when both domains are full, else one scan over base.
    /// Valid for every descendant state of the worker.
    fn wide_seed(&mut self, sh: &Shared, l: usize, r: usize, ci: usize, full: &Bits) {
        let tb = if self.dom[l] == *full && self.dom[r] == *full {
            let t = sh.table(l, r);
            TermBox {
                have: true,
                snap_l: *full,
                snap_r: *full,
                lo_re: t.flo_re,
                hi_re: t.fhi_re,
                lo_im: t.flo_im,
                hi_im: t.fhi_im,
            }
        } else {
            self.term_box_tight(sh, l, r, ci)
        };
        self.wcache[ci] = tb;
        self.tcache[ci] = tb;
    }

    /// Best sound box for the check: tight when valid, else the wide seed
    /// (always valid within a worker, since domains only shrink from base).
    fn term_box_for_check(&self, l: usize, r: usize, ci: usize) -> (TermBox, bool) {
        let t = &self.tcache[ci];
        if t.valid_for(&self.dom[l], &self.dom[r]) {
            return (*t, true);
        }
        debug_assert!(self.wcache[ci].have, "wide seed must exist before checks");
        (self.wcache[ci], false)
    }

    /// One block equation: sum_bk slot(bi,bk)*slot(bk,bj) = target (2x2).
    /// Box check always; exact residual arc-consistency when one term remains.
    /// Returns false on certain failure; shrunk slots are queued for their
    /// own incident equations.
    fn eq_check(&mut self, sh: &Shared, bi: usize, bj: usize) -> bool {
        let m = sh.m;
        let lam = K::lambda();
        if let Some(live) = &self.live {
            live.eq_checks.fetch_add(1, Ordering::Relaxed);
        }
        // Exact assigned partial sum per entry, and the unassigned term list.
        let mut sum: [K; 4] = [K::zero(); 4];
        let mut un: [usize; 3] = [0; 3];
        let mut nun = 0usize;
        for bk in 0..m {
            let (l, r) = (bi * m + bk, bk * m + bj);
            let sl = b_singleton(&self.dom[l]);
            let sr = b_singleton(&self.dom[r]);
            let assigned = if l == r { sl.is_some() } else { sl.is_some() && sr.is_some() };
            if assigned {
                let p = if l == r {
                    let x = &sh.slots[l][sl.unwrap()];
                    mul22(x, x)
                } else {
                    mul22(&sh.slots[l][sl.unwrap()], &sh.slots[r][sr.unwrap()])
                };
                for u in 0..2 {
                    for w in 0..2 {
                        let e = 2 * u + w;
                        sum[e] = sum[e].add(p[u][w]);
                    }
                }
            } else {
                un[nun] = bk;
                nun += 1;
            }
        }
        // Box feasibility per scalar entry. Best sound box per unassigned
        // term: tight when its snapshot covers the current domains, else the
        // worker-wide box (always valid in-worker). A wide-box failure is a
        // sound prune (wide is a superset of the achievable set). When the
        // wide box passes and a tight rescan is cheap, tighten for power.
        let mut tbs: [TermBox; 3] = [TermBox::empty(); 3];
        let mut was_tight = [false; 3];
        for (i, &bk) in un.iter().take(nun).enumerate() {
            let (l, r) = (bi * m + bk, bk * m + bj);
            let ci = (bi * m + bj) * m + bk;
            let (tb, tight) = self.term_box_for_check(l, r, ci);
            tbs[i] = tb;
            was_tight[i] = tight;
        }
        if !bracket_ok(&sum, &tbs[..nun], bi, bj, lam) {
            return false;
        }
        for (i, &bk) in un.iter().take(nun).enumerate() {
            if was_tight[i] {
                continue;
            }
            let (l, r) = (bi * m + bk, bk * m + bj);
            let cost = if l == r {
                b_popc(&self.dom[l]) as usize
            } else {
                b_popc(&self.dom[l]) as usize * b_popc(&self.dom[r]) as usize
            };
            if cost > tight_max_pairs() {
                continue;
            }
            let ci = (bi * m + bj) * m + bk;
            tbs[i] = self.term_box_tight(sh, l, r, ci);
            if !bracket_ok(&sum, &tbs[..nun], bi, bj, lam) {
                return false;
            }
        }
        if nun == 0 {
            // Fully assigned: exact equality (this fires at leaves; the
            // residual filters below normally force it earlier).
            for u in 0..2 {
                for w in 0..2 {
                    let e = 2 * u + w;
                    let target = if bi == bj && u == w { lam } else { K::zero() };
                    if sum[e] != target {
                        return false;
                    }
                }
            }
            return true;
        }
        if nun > 1 {
            return true;
        }
        // Exact residual: the remaining term's product must equal R exactly.
        let bk = un[0];
        let (l, r) = (bi * m + bk, bk * m + bj);
        let mut resid: K2 = [[K::zero(); 2]; 2];
        for u in 0..2 {
            for w in 0..2 {
                let e = 2 * u + w;
                let target = if bi == bj && u == w { lam } else { K::zero() };
                resid[u][w] = target.sub(sum[e]);
            }
        }
        let dl = self.dom[l];
        let dr = self.dom[r];
        // Stage 1: product-to-support map lookup. xset[R] holds every left
        // value having SOME partner over the full slot lists with product R,
        // so intersecting with the current domains is sound: a value with no
        // partner anywhere is removed. A miss means R is unachievable.
        // One hash lookup plus two 13-word ANDs replaces the old
        // per-candidate exact inverse solves (the m=3 hotspot).
        if let Some(live) = &self.live {
            live.stage1.fetch_add(1, Ordering::Relaxed);
        }
        let (xs, ys) = match sh.pmap(l, r).get(&resid) {
            Some(xy) => *xy,
            None => {
                if let Some(live) = &self.live {
                    live.stage1_miss.fetch_add(1, Ordering::Relaxed);
                }
                return false;
            }
        };
        let mut nl = [0u64; SW];
        let mut nr = [0u64; SW];
        for w in 0..SW {
            nl[w] = xs[w] & dl[w];
            nr[w] = ys[w] & dr[w];
        }
        if b_empty(&nl) || b_empty(&nr) {
            return false;
        }
        if l != r {
            // Stage 2: refine to full pairwise arc consistency (a value is
            // kept only with a partner inside the CURRENT domains) when the
            // candidate rectangle is small. Stage 1 alone keeps values whose
            // partners all lie outside the current domains; stage 2 removes
            // those. Square terms need no stage 2: their map records squares
            // only, so stage 1 is already exact. f64 prefilter against the
            // residual, exact mul22 confirmation for candidates.
            let cl = b_popc(&nl) as usize;
            let cr = b_popc(&nr) as usize;
            if cl * cr <= refine_max_pairs() {
                if let Some(live) = &self.live {
                    live.stage2.fetch_add(1, Ordering::Relaxed);
                }
                let t = sh.table(l, r);
                let mut rre = [0f64; 4];
                let mut rim = [0f64; 4];
                for u in 0..2 {
                    for w in 0..2 {
                        let (rr, ri) = kf64(resid[u][w]);
                        rre[2 * u + w] = rr;
                        rim[2 * u + w] = ri;
                    }
                }
                let close = |a: &[f64; 4], b: &[f64; 4]| -> bool {
                    (0..4).all(|e| (a[e] - b[e]).abs() <= 1e-9 * (1.0 + b[e].abs()))
                };
                // Right support: y survives iff some x in nl hits R.
                let mut sup_r = [0u64; SW];
                b_each(&nl, |x| {
                    let base = x * t.ny;
                    b_each(&nr, |y| {
                        if b_test(&sup_r, y) {
                            return;
                        }
                        let i = base + y;
                        if close(&t.re[i], &rre)
                            && close(&t.im[i], &rim)
                            && mul22(&sh.slots[l][x], &sh.slots[r][y]) == resid
                        {
                            b_set(&mut sup_r, y);
                        }
                    });
                });
                // Left support: x survives iff some y in sup_r hits R.
                // (Any witness pair has both endpoints globally supported,
                // so refining against sup_r loses nothing.)
                let mut sup_l = [0u64; SW];
                b_each(&sup_r, |y| {
                    b_each(&nl, |x| {
                        if b_test(&sup_l, x) {
                            return;
                        }
                        let i = x * t.ny + y;
                        if close(&t.re[i], &rre)
                            && close(&t.im[i], &rim)
                            && mul22(&sh.slots[l][x], &sh.slots[r][y]) == resid
                        {
                            b_set(&mut sup_l, x);
                        }
                    });
                });
                nl = sup_l;
                nr = sup_r;
                if b_empty(&nl) || b_empty(&nr) {
                    return false;
                }
            }
        }
        if nl != dl {
            self.dom[l] = nl;
            self.mark_dirty(l);
        }
        if nr != dr {
            self.dom[r] = nr;
            self.mark_dirty(r);
        }
        true
    }

    /// trace(g) = 0 bracket: assigned diagonal-slot traces plus per-slot
    /// min/max over the surviving domain must bracket 0 in both parts.
    fn trace_check(&mut self, sh: &Shared) -> bool {
        let m = sh.m;
        let (mut lo_re, mut hi_re) = (0f64, 0f64);
        let (mut lo_im, mut hi_im) = (0f64, 0f64);
        for bi in 0..m {
            let s = bi * m + bi;
            if let Some(x) = b_singleton(&self.dom[s]) {
                let r = sh.tr_re[bi][x];
                let i = sh.tr_im[bi][x];
                lo_re += r - pad(r);
                hi_re += r + pad(r);
                lo_im += i - pad(i);
                hi_im += i + pad(i);
            } else {
                let (mut mn_r, mut mx_r) = (f64::INFINITY, f64::NEG_INFINITY);
                let (mut mn_i, mut mx_i) = (f64::INFINITY, f64::NEG_INFINITY);
                b_each(&self.dom[s], |x| {
                    mn_r = mn_r.min(sh.tr_re[bi][x]);
                    mx_r = mx_r.max(sh.tr_re[bi][x]);
                    mn_i = mn_i.min(sh.tr_im[bi][x]);
                    mx_i = mx_i.max(sh.tr_im[bi][x]);
                });
                lo_re += mn_r - pad(mn_r);
                hi_re += mx_r + pad(mx_r);
                lo_im += mn_i - pad(mn_i);
                hi_im += mx_i + pad(mx_i);
            }
        }
        lo_re <= 0.0 && hi_re >= 0.0 && lo_im <= 0.0 && hi_im >= 0.0
    }

    /// Propagate to the greatest fixpoint via the dirty-slot queue: only
    /// equations incident to a shrunk slot are re-checked. Same fixpoint as
    /// full scanning (monotone operators, chaotic iteration), much cheaper.
    fn propagate(&mut self, sh: &Shared) -> bool {
        while let Some(p) = self.queue.pop() {
            if let Some(live) = &self.live {
                live.props.fetch_add(1, Ordering::Relaxed);
            }
            self.dirty[p] = false;
            if self.dom.iter().any(b_empty) {
                return false;
            }
            let (a, b) = (p / sh.m, p % sh.m);
            // Slot (a,b) is the left factor of term k=b in equations (a, bj)
            // and the right factor of term k=a in equations (bi, b).
            for bj in 0..sh.m {
                if !self.eq_check(sh, a, bj) {
                    return false;
                }
            }
            for bi in 0..sh.m {
                if bi != a && !self.eq_check(sh, bi, b) {
                    return false;
                }
            }
            if a == b && !self.trace_check(sh) {
                return false;
            }
        }
        !self.dom.iter().any(b_empty)
    }

    /// Leaf: assemble g, verify g^2 = lambda I exactly, map back to integer G,
    /// verify G^2 = A, accumulate commutatively.
    fn leaf(&mut self, sh: &Shared) {
        if let Some(live) = &self.live {
            live.leaves.fetch_add(1, Ordering::Relaxed);
        }
        let n = sh.n;
        let m = sh.m;
        let mut g = vec![vec![K::zero(); n]; n];
        for bi in 0..m {
            for bj in 0..m {
                let x = b_singleton(&self.dom[bi * m + bj]).expect("leaf domains are singletons");
                let slot = &sh.slots[bi * m + bj][x];
                for u in 0..2 {
                    for w in 0..2 {
                        g[2 * bi + u][2 * bj + w] = slot[u][w];
                    }
                }
            }
        }
        let lam = K::lambda();
        for i in 0..n {
            for j in 0..n {
                let mut acc = K::zero();
                for k in 0..n {
                    acc = acc.add(g[i][k].mul(g[k][j]));
                }
                let target = if i == j { lam } else { K::zero() };
                if acc != target {
                    return; // cannot happen after exact propagation; stay safe
                }
            }
        }
        let Some(gi) = int_of_g(&sh.coords, &g) else {
            return;
        };
        debug_assert!(imm(&gi, &gi) == sh.coords.a, "mapped G must square to A");
        let h = hash_intmat(&gi);
        self.count += 1;
        if let Some(live) = &self.live {
            live.solutions.fetch_add(1, Ordering::Relaxed);
        }
        self.checksum = self.checksum.wrapping_add(h);
        let key = class_key(&gi, m);
        match self.hist.entry(key) {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                let r = e.get_mut();
                r.count += 1;
                r.checksum = r.checksum.wrapping_add(h);
                if h < r.rep_hash {
                    r.rep_hash = h;
                    r.rep = gi.clone();
                }
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(ClassRec {
                    count: 1,
                    checksum: h,
                    rep_hash: h,
                    rep: gi.clone(),
                });
            }
        }
        if self.samples.len() < 8 {
            self.samples.push(gi);
        }
    }

    fn recurse(&mut self, sh: &Shared, cap: Option<u64>) {
        if let Some(c) = cap {
            if self.count >= c {
                return;
            }
        }
        // MRV: smallest surviving domain bigger than one.
        let mut pick = None;
        let mut best = u32::MAX;
        for s in 0..sh.m * sh.m {
            let p = b_popc(&self.dom[s]);
            if p > 1 && p < best {
                best = p;
                pick = Some(s);
            }
        }
        self.nodes += 1;
        if let Some(live) = &self.live {
            live.nodes.fetch_add(1, Ordering::Relaxed);
            live.depth_hist[(self.depth as usize).min(31)].fetch_add(1, Ordering::Relaxed);
            let w = &live.workers[self.wid];
            w.nodes.fetch_add(1, Ordering::Relaxed);
            w.depth.store(self.depth as u64, Ordering::Relaxed);
            w.max_depth.fetch_max(self.depth as u64, Ordering::Relaxed);
        }
        let Some(s) = pick else {
            self.leaf(sh);
            return;
        };
        let save_dom = self.dom.clone();
        let mut xs = Vec::new();
        b_each(&self.dom[s], |x| xs.push(x));
        for x in xs {
            self.dom[s] = [0u64; SW];
            b_set(&mut self.dom[s], x);
            self.mark_dirty(s);
            if self.propagate(sh) {
                self.depth += 1;
                self.recurse(sh, cap);
                self.depth -= 1;
            }
            // Backtrack: restore domains and the (empty-at-branch-time)
            // propagation queue. Tight boxes from the abandoned branch fail
            // their snapshot-subset test automatically; wide boxes stay
            // valid throughout.
            self.dom.clone_from_slice(&save_dom);
            self.queue.clear();
            self.dirty.fill(false);
            if let Some(c) = cap {
                if self.count >= c {
                    return;
                }
            }
        }
    }
}

/// Per-prefix immutable result: the atomic unit of the stratified benchmark
/// and (later) the durable shard format. Commutative under count/checksum sum.
pub struct ItemStat {
    pub item: u32,
    pub count: u64,
    pub checksum: u64,
    pub nodes: u64,
    pub seconds: f64,
}

fn run_worker(
    sh: Arc<Shared>,
    base_dom: &[Bits],
    x0: usize,
    cap: Option<u64>,
    live: Option<Arc<LiveStats>>,
    wid: usize,
) -> (ItemStat, Vec<IntMat>, u64, HashMap<ClassKey, ClassRec>) {
    let t0 = Instant::now();
    let mut s = Search::new(&sh);
    if let Some(l) = &live {
        l.item_start(wid, x0 as u32);
    }
    s.live = live.clone();
    s.wid = wid;
    s.dom.clone_from_slice(base_dom);
    s.dom[0] = [0u64; SW];
    b_set(&mut s.dom[0], x0);
    // Seed wide boxes once; they stay valid for the worker's whole subtree.
    let full = b_full(sh.slots[0].len());
    let m = sh.m;
    for bi in 0..m {
        for bj in 0..m {
            for bk in 0..m {
                let (l, r) = (bi * m + bk, bk * m + bj);
                let ci = (bi * m + bj) * m + bk;
                s.wide_seed(&sh, l, r, ci, &full);
            }
        }
    }
    s.mark_dirty(0);
    if s.propagate(&sh) {
        s.recurse(&sh, cap);
    }
    (
        ItemStat {
            item: x0 as u32,
            count: s.count,
            checksum: s.checksum,
            nodes: s.nodes,
            seconds: t0.elapsed().as_secs_f64(),
        },
        s.samples,
        s.tight_computes,
        s.hist,
    )
}

pub struct CspStats {
    pub count: u64,
    pub checksum: u64,
    pub nodes: u64,
    pub seconds: f64,
    pub samples: Vec<IntMat>,
    pub complete: bool,
    pub shared_table: bool,
    pub items: usize,
    pub items_detail: Vec<ItemStat>,
    pub table_bytes: usize,
    pub tight_computes: u64,
    pub hist: HashMap<ClassKey, ClassRec>,
}

/// Enumerate with the CSP engine. Work items are the surviving values of
/// slot 0 after root propagation; `stride > 1` processes every stride-th item
/// (a stratified sample for benchmarking; results are then a SAMPLE, marked
/// complete=false).
pub fn solve(sh: Arc<Shared>, threads: usize, cap: Option<u64>, stride: usize) -> CspStats {
    let t0 = Instant::now();
    // Root propagation on full domains gives the base state for all workers.
    let mut root = Search::new(&sh);
    let table_bytes: usize = sh
        .tables
        .iter()
        .flatten()
        .map(|t| t.re.len() * 4 * 8 * 2)
        .sum();
    // Root wide seeds: all domains full, so every term gets the precomputed
    // full-domain box for free.
    let full = b_full(sh.slots[0].len());
    for bi in 0..sh.m {
        for bj in 0..sh.m {
            for bk in 0..sh.m {
                let (l, r) = (bi * sh.m + bk, bk * sh.m + bj);
                let ci = (bi * sh.m + bj) * sh.m + bk;
                root.wide_seed(&sh, l, r, ci, &full);
            }
        }
    }
    for p in 0..sh.m * sh.m {
        root.mark_dirty(p);
    }
    if !root.propagate(&sh) {
        return CspStats {
            count: 0,
            checksum: 0,
            nodes: 0,
            seconds: t0.elapsed().as_secs_f64(),
            samples: Vec::new(),
            complete: true,
            shared_table: sh.shared_table,
            items: 0,
            items_detail: Vec::new(),
            table_bytes,
            tight_computes: 0,
            hist: HashMap::new(),
        };
    }
    let mut items = Vec::new();
    b_each(&root.dom[0], |x| items.push(x));
    let total = items.len();
    if stride > 1 {
        items = items.into_iter().step_by(stride).collect();
    }
    println!(
        "csp: shared_table={} items={} (of {} slot-0 values, stride {})",
        sh.shared_table,
        items.len(),
        total,
        stride
    );
    let base_dom = root.dom.clone();
    let next = Arc::new(AtomicU64::new(0));
    let done = Arc::new(AtomicBool::new(false));
    let live = Arc::new(LiveStats::new(
        items.len() as u64,
        threads.max(1).min(items.len().max(1)),
        sh.m,
    ));
    let hb = spawn_heartbeat(
        Arc::clone(&live),
        // Per-process suffix, same convention as run_shards: two concurrent
        // processes sharing the env var must not interleave appends.
        std::env::var("ADINKRA_CSP_STATUS")
            .ok()
            .map(|p| format!("{p}.{}.{}", live.pod, std::process::id())),
        Arc::clone(&done),
    );
    let (mut count, mut checksum, mut nodes) = (0u64, 0u64, 0u64);
    let mut tight_computes = 0u64;
    let mut hist: HashMap<ClassKey, ClassRec> = HashMap::new();
    let mut samples = Vec::new();
    let mut items_detail: Vec<ItemStat> = Vec::new();
    let nitems = items.len();
    std::thread::scope(|scope| {
        let mut hs = Vec::new();
        for wid in 0..threads.max(1).min(nitems.max(1)) {
            let sh = Arc::clone(&sh);
            let next = Arc::clone(&next);
            let live = Arc::clone(&live);
            let base_dom = &base_dom;
            let items = &items;
            hs.push(scope.spawn(move || {
                let mut stats = Vec::new();
                let mut smp = Vec::new();
                let mut tc = 0u64;
                let mut hh: HashMap<ClassKey, ClassRec> = HashMap::new();
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i as usize >= items.len() {
                        break;
                    }
                    let (st, ss, tt, wh) = run_worker(
                        Arc::clone(&sh),
                        base_dom,
                        items[i as usize],
                        cap,
                        Some(Arc::clone(&live)),
                        wid,
                    );
                    eprintln!(
                        "item {:04} done: count={} nodes={} {:.1}s",
                        st.item, st.count, st.nodes, st.seconds
                    );
                    stats.push(st);
                    smp.extend(ss);
                    tc += tt;
                    merge_hist(&mut hh, wh);
                    // Completion is counted by the caller now that run_worker
                    // no longer does it: here the item's results are already
                    // folded into the worker-local accumulators.
                    live.item_done(wid);
                }
                (stats, smp, tc, hh)
            }));
        }
        for h in hs {
            let (stats, smp, tc, hh) = h.join().unwrap();
            tight_computes += tc;
            merge_hist(&mut hist, hh);
            for st in stats {
                count += st.count;
                checksum = checksum.wrapping_add(st.checksum);
                nodes += st.nodes;
                items_detail.push(st);
            }
            samples.extend(smp);
        }
    });
    done.store(true, Ordering::Relaxed);
    // Final synchronous record: the sleeping heartbeat thread is killed at
    // process exit, so without this a run shorter than one interval would
    // leave no status output at all.
    if let Some(h) = &hb {
        h.emit();
    }
    items_detail.sort_by_key(|st| st.item);
    samples.truncate(8);
    CspStats {
        count,
        checksum,
        nodes,
        seconds: t0.elapsed().as_secs_f64(),
        samples,
        complete: cap.is_none() && stride == 1,
        shared_table: sh.shared_table,
        items: nitems,
        items_detail,
        table_bytes,
        tight_computes,
        hist,
    }
}

// ===========================================================================
// Durable distributed sharding: one immutable shard per slot-0 prefix value
// ===========================================================================

const ENGINE_VERSION: &str = "gmatrix_csp.v2-ac3";

fn mat_json(g: &IntMat) -> String {
    let rows: Vec<String> = g
        .iter()
        .map(|r| format!("[{}]", r.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",")))
        .collect();
    format!("[{}]", rows.join(","))
}

/// Total deterministic order over classes for JSON emission: count desc,
/// then the ClassKey fields asc (two classes equal on nnz/support differ in
/// ranks, else they would be the same key), rep_hash as the final guard, so
/// emitted JSON never depends on HashMap iteration order.
fn class_order(a: &(&ClassKey, &ClassRec), b: &(&ClassKey, &ClassRec)) -> std::cmp::Ordering {
    b.1.count
        .cmp(&a.1.count)
        .then(a.0.nnz.cmp(&b.0.nnz))
        .then(a.0.support.cmp(&b.0.support))
        .then(a.0.ranks.cmp(&b.0.ranks))
        .then(a.1.rep_hash.cmp(&b.1.rep_hash))
}

fn classes_to_json(hist: &HashMap<ClassKey, ClassRec>, m: usize) -> String {
    let mut classes: Vec<(&ClassKey, &ClassRec)> = hist.iter().collect();
    classes.sort_by(class_order);
    let parts: Vec<String> = classes
        .iter()
        .map(|(k, r)| {
            format!(
                "{{\"nnz\":{},\"support\":{},\"ranks\":[{}],\"count\":{},\"checksum\":\"{:016x}\",\"rep_hash\":\"{:016x}\",\"rep\":{}}}",
                k.nnz,
                k.support,
                k.ranks
                    .iter()
                    .take(m * m)
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
                r.count,
                r.checksum,
                r.rep_hash,
                mat_json(&r.rep)
            )
        })
        .collect();
    format!("[{}]", parts.join(","))
}

fn shard_path(dir: &str, item: u32) -> String {
    format!("{dir}/shard_{item:04}.json")
}

/// A fully decoded and validated shard: identity checked, every class
/// semantically re-verified. This is the single validation gate shared by
/// resume (run_shards), the status dashboard, and the merge; a shard that
/// passes here is exactly what the census may aggregate.
struct ShardData {
    count: u64,
    checksum: u64,
    nodes: u64,
    seconds: f64,
    classes: Vec<(ClassKey, ClassRec)>,
}

/// Decode and validate an already-parsed shard value. Checks, in order:
/// identity fields; finite nonnegative seconds; 16-hex checksums; the
/// slot-0 binding (below); exactly m*m ranks and a (4m)x(4m) rep with
/// entries in {-1,0,1} per class, all decoded with try_from (no narrowing
/// casts); the semantic census invariants (rep squares to A, rep_hash ==
/// hash_intmat(rep), declared nnz/support/ranks == class_key(rep)); class
/// counts summing to the shard count without overflow; class checksums
/// wrap-summing to the shard checksum. Returns None on any violation;
/// never panics.
///
/// Slot-0 binding: item indexes the canonical slot-0 alphabet `slot0`, and
/// every solution under that prefix has g's slot (0,0) equal to
/// slot0[item], so every class rep's leading 4x4 integer block must be
/// exactly that value's image under slot_int_of_g. The map is injective on
/// the alphabet (conjugation by the invertible per-block change of basis),
/// so a shard copied under a different item can never validate.
fn shard_decode_value(
    v: &serde_json::Value,
    side: Side,
    m: usize,
    item: u64,
    coords: &Coords,
    slot0: &[K2],
) -> Option<ShardData> {
    if !(v["engine"].as_str() == Some(ENGINE_VERSION)
        && v["side"].as_str() == Some(side.name())
        && v["blocks"].as_u64() == Some(m as u64)
        && v["item"].as_u64() == Some(item)
        && v["complete"].as_bool() == Some(true))
    {
        return None;
    }
    // try_from instead of `as usize`: on a 32-bit target a forged item
    // >= 2^32 must refuse, never alias a small slot index.
    let want0 = slot_int_of_g(coords, 0, 0, slot0.get(usize::try_from(item).ok()?)?)?;
    let count = v["count"].as_u64()?;
    let nodes = v["nodes"].as_u64()?;
    let seconds = v["seconds"]
        .as_f64()
        .filter(|s| s.is_finite() && *s >= 0.0)?;
    let hex16 = |c: &serde_json::Value| -> Option<u64> {
        let s = c.as_str().filter(|s| s.len() == 16)?;
        u64::from_str_radix(s, 16).ok()
    };
    let checksum = hex16(&v["checksum"])?;
    let classes_v = v["classes"].as_array()?;
    let mut class_sum = 0u64;
    let mut class_csum = 0u64;
    let mut classes = Vec::with_capacity(classes_v.len());
    for c in classes_v {
        let ccount = c["count"].as_u64()?;
        class_sum = class_sum.checked_add(ccount)?;
        let nnz = u32::try_from(c["nnz"].as_u64()?).ok()?;
        let support = u16::try_from(c["support"].as_u64()?).ok()?;
        let rs = c["ranks"].as_array()?;
        if rs.len() != m * m {
            return None;
        }
        let mut ranks = [0xFFu8; 9];
        for (i, r) in rs.iter().enumerate() {
            ranks[i] = u8::try_from(r.as_u64()?).ok()?;
        }
        let cchecksum = hex16(&c["checksum"])?;
        class_csum = class_csum.wrapping_add(cchecksum);
        let rep_hash = hex16(&c["rep_hash"])?;
        let rows = c["rep"].as_array()?;
        if rows.len() != 4 * m {
            return None;
        }
        let mut rep: IntMat = Vec::with_capacity(4 * m);
        for row in rows {
            let xs = row.as_array()?;
            if xs.len() != 4 * m {
                return None;
            }
            let mut r: Vec<i32> = Vec::with_capacity(4 * m);
            for x in xs {
                let e = x.as_i64()?;
                if !(-1..=1).contains(&e) {
                    return None;
                }
                r.push(e as i32);
            }
            rep.push(r);
        }
        // Slot-0 binding: the rep's leading 4x4 block must be exactly the
        // integer image of this shard's claimed slot-0 prefix value.
        for i in 0..4 {
            if rep[i][..4] != want0[i][..] {
                return None;
            }
        }
        // Semantic re-verification: the rep must square to A, and every
        // declared field must be reproducible from the rep's contents.
        if imm(&rep, &rep) != coords.a {
            return None;
        }
        if hash_intmat(&rep) != rep_hash {
            return None;
        }
        let want = class_key(&rep, m);
        if want.nnz != nnz || want.support != support || want.ranks != ranks {
            return None;
        }
        classes.push((
            ClassKey {
                nnz,
                support,
                ranks,
            },
            ClassRec {
                count: ccount,
                checksum: cchecksum,
                rep_hash,
                rep,
            },
        ));
    }
    if class_sum != count || class_csum != checksum {
        return None;
    }
    Some(ShardData {
        count,
        checksum,
        nodes,
        seconds,
        classes,
    })
}

/// Read, parse, and fully validate one shard file (single read: no
/// validate-then-reread window).
fn shard_decode(
    path: &str,
    side: Side,
    m: usize,
    item: u64,
    coords: &Coords,
    slot0: &[K2],
) -> Option<ShardData> {
    let text = std::fs::read_to_string(path).ok()?;
    let v = serde_json::from_str::<serde_json::Value>(&text).ok()?;
    shard_decode_value(&v, side, m, item, coords, slot0)
}

/// Validation context for shard decoding without the expensive pair tables:
/// the coordinate change (target matrix A and the per-block intertwining
/// maps) plus the canonical slot-0 alphabet that item indexes. Status builds
/// this per (side, blocks); run_shards/run_merge take the same pieces from
/// the already-built Shared instead.
fn shard_ctx(side: Side, m: usize) -> (Coords, Vec<K2>) {
    let coords = build_coords(m, side);
    let (_, blocks) = cls_a_blocks(m, side);
    let alph = build_alphabets(&coords, &blocks);
    let slot0 = alph.slots.into_iter().next().unwrap_or_default();
    (coords, slot0)
}

/// Install one shard file durably: the caller has written a pod+pid-unique
/// tmp, then an atomic no-replace hard_link publishes it. A lost race is
/// accepted when the winner's file fully decodes; a corrupt or partial
/// existing file is replaced by rename (resume re-runs any item without a
/// valid shard, so a stale corrupt file must never block a fresh write).
/// Filesystems without hard links fall back to rename plus a post-install
/// decode. Returns true only when `path` holds a fully valid shard for this
/// item afterwards.
fn install_shard(tmp: &str, path: &str, side: Side, m: usize, item: u64, sh: &Shared) -> bool {
    let decodes = || shard_decode(path, side, m, item, &sh.coords, &sh.slots[0]).is_some();
    match std::fs::hard_link(tmp, path) {
        Ok(()) => {
            let _ = std::fs::remove_file(tmp);
            true
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            if decodes() {
                let _ = std::fs::remove_file(tmp);
                true
            } else if std::fs::rename(tmp, path).is_ok() {
                decodes()
            } else {
                let _ = std::fs::remove_file(tmp);
                false
            }
        }
        // No hard-link support (or cross-device): rename install, verify.
        Err(_) => {
            if std::fs::rename(tmp, path).is_ok() {
                decodes()
            } else {
                let _ = std::fs::remove_file(tmp);
                false
            }
        }
    }
}

fn shard_json(
    side: Side,
    m: usize,
    st: &ItemStat,
    hist: &HashMap<ClassKey, ClassRec>,
) -> String {
    format!(
        "{{\n  \"shard_id\": \"cls-g-csp-{}-m{}-item{:04}\",\n  \"engine\": \"{}\",\n  \"side\": \"{}\",\n  \"blocks\": {},\n  \"item\": {},\n  \"count\": {},\n  \"checksum\": \"{:016x}\",\n  \"nodes\": {},\n  \"seconds\": {:.3},\n  \"complete\": true,\n  \"classes\": {}\n}}\n",
        side.name(),
        m,
        st.item,
        ENGINE_VERSION,
        side.name(),
        m,
        st.item,
        st.count,
        st.checksum,
        st.nodes,
        st.seconds,
        classes_to_json(hist, m)
    )
}

/// Everything derivable from (side, m): the built tables, the propagated
/// root, and the canonical work-item set (the surviving slot-0 values after
/// root propagation). The canonical item set is the definition of full
/// coverage; the persisted items.json manifest must equal it exactly.
struct Canonical {
    sh: Arc<Shared>,
    root: Search,
    items: Vec<usize>,
}

fn canonical_run(side: Side, m: usize) -> Canonical {
    let coords = build_coords(m, side);
    let (_, blocks) = cls_a_blocks(m, side);
    let alph = build_alphabets(&coords, &blocks);
    let sh = Arc::new(build_shared(coords, &alph));
    let mut root = Search::new(&sh);
    let full = b_full(sh.slots[0].len());
    for bi in 0..sh.m {
        for bj in 0..sh.m {
            for bk in 0..sh.m {
                let (l, r) = (bi * sh.m + bk, bk * sh.m + bj);
                let ci = (bi * sh.m + bj) * sh.m + bk;
                root.wide_seed(&sh, l, r, ci, &full);
            }
        }
    }
    for p in 0..sh.m * sh.m {
        root.mark_dirty(p);
    }
    let feasible = root.propagate(&sh);
    let mut items = Vec::new();
    if feasible {
        b_each(&root.dom[0], |x| items.push(x));
    }
    Canonical { sh, root, items }
}

/// Remove crash-leaked tmp files older than one hour. Live tmps are
/// written and installed within seconds and carry pod+pid-unique names, so
/// age is the safe liveness proxy: a writer whose tmp is older than the
/// cutoff is gone. Only regular files with engine-produced tmp names are
/// candidates; anything else in the directory is not ours to delete.
fn sweep_stale_tmps(dir: &str) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let cutoff = std::time::SystemTime::now() - std::time::Duration::from_secs(3600);
    for e in rd.flatten() {
        let Ok(ft) = e.file_type() else {
            continue;
        };
        if !ft.is_file() {
            continue;
        }
        // The only tmp names this engine produces in a shard directory:
        // shard_XXXX.json.tmp.<pod>.<pid>, items.json.tmp.<pod>.<pid>, and
        // status_<pod>.<pid>.json.tmp (snapshot publish). Merge-output tmps
        // live next to the caller-chosen output path, not necessarily here.
        let name = e.file_name().to_string_lossy().to_string();
        let ours = (name.starts_with("shard_") && name.contains(".json.tmp."))
            || name.starts_with("items.json.tmp.")
            || (name.starts_with("status_") && name.ends_with(".tmp"));
        if !ours {
            continue;
        }
        let stale = e
            .metadata()
            .and_then(|md| md.modified())
            .map(|t| t < cutoff)
            .unwrap_or(false);
        if stale {
            let _ = std::fs::remove_file(e.path());
        }
    }
}

/// Run items [start, start+count) of the slot-0 prefix list, writing one
/// immutable shard JSON per completed item (unique tmp + atomic no-replace
/// install). Existing valid shards are skipped, so reruns resume exactly
/// where they stopped. Returns false only on a fatal configuration
/// conflict (an established manifest with different contents).
pub fn run_shards(
    side: Side,
    m: usize,
    start: usize,
    count: usize,
    threads: usize,
    dir: &str,
    stride: usize,
) -> bool {
    println!(
        "cls-g-csp-shard: side={} blocks={} start={} count={} threads={} dir={} stride={}",
        side.name(),
        m,
        start,
        count,
        threads,
        dir,
        stride
    );
    if let Err(e) = std::fs::create_dir_all(dir) {
        eprintln!("cannot create shard dir {dir}: {e}");
        return false;
    }
    sweep_stale_tmps(dir);
    let t_build = Instant::now();
    let canon = canonical_run(side, m);
    let sh = Arc::clone(&canon.sh);
    let build_secs = t_build.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let table_bytes: usize = sh
        .tables
        .iter()
        .flatten()
        .map(|t| t.re.len() * 4 * 8 * 2)
        .sum();
    println!("tables built: {} bytes in {:.1}s", table_bytes, build_secs);
    println!(
        "config: tight_max_pairs={} refine_max_pairs={} heartbeat_s={} status={}/status_<pod>.<pid>.jsonl",
        tight_max_pairs(),
        refine_max_pairs(),
        heartbeat_secs(),
        dir
    );
    let items = &canon.items;
    if items.is_empty() {
        println!("root infeasible; nothing to do");
        return true;
    }
    // Persist the item list for the merge step's coverage verification.
    // Fleet-safe publish: the pre-check is only a fast path; the arbiter is
    // an atomic no-replace hard_link from a pod+pid-unique tmp, so no
    // interleaving of concurrent first-creators can clobber a manifest. An
    // identical established manifest is accepted; a conflicting one aborts
    // the run (nonzero exit at the CLI).
    let items_json = format!(
        "{{\"engine\":\"{}\",\"side\":\"{}\",\"blocks\":{},\"items\":[{}]}}\n",
        ENGINE_VERSION,
        side.name(),
        m,
        items.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",")
    );
    let items_path = format!("{dir}/items.json");
    let pod = pod_name();
    let manifest_ok = match std::fs::read_to_string(&items_path) {
        Ok(existing) => existing == items_json,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let tmp = format!("{dir}/items.json.tmp.{pod}.{}", std::process::id());
            if let Err(e) = std::fs::write(&tmp, &items_json) {
                eprintln!("cannot write {tmp}: {e}");
                return false;
            }
            match std::fs::hard_link(&tmp, &items_path) {
                Ok(()) => {
                    let _ = std::fs::remove_file(&tmp);
                    true
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    let _ = std::fs::remove_file(&tmp);
                    // Lost the race: accept the winner only when its contents
                    // are exactly what this process would have published.
                    std::fs::read_to_string(&items_path).ok().as_deref()
                        == Some(items_json.as_str())
                }
                Err(_) => {
                    // Filesystem without hard links: rename install, then
                    // verify the landed bytes (best effort; a concurrent
                    // publisher's conflicting bytes fail the comparison).
                    if std::fs::rename(&tmp, &items_path).is_ok() {
                        std::fs::read_to_string(&items_path).ok().as_deref()
                            == Some(items_json.as_str())
                    } else {
                        let _ = std::fs::remove_file(&tmp);
                        false
                    }
                }
            }
        }
        Err(e) => {
            // Present but unreadable is not "absent": never publish over a
            // manifest this process could not inspect.
            eprintln!("cannot read {items_path}: {e}");
            return false;
        }
    };
    if !manifest_ok {
        eprintln!(
            "items.json already exists in {dir} with different contents; refusing to overwrite an established manifest"
        );
        return false;
    }
    let end = start.saturating_add(count).min(items.len());
    if start >= items.len() {
        println!("start {} beyond item list length {}", start, items.len());
        return true;
    }
    let todo: Vec<usize> = items[start..end]
        .iter()
        .step_by(stride.max(1))
        .copied()
        .filter(|&x| {
            shard_decode(&shard_path(dir, x as u32), side, m, x as u64, &sh.coords, &sh.slots[0])
                .is_none()
        })
        .collect();
    println!(
        "items {}..{} stride {} of {}; {} already sharded, {} to run",
        start,
        end,
        stride,
        items.len(),
        (end - start).div_ceil(stride.max(1)) - todo.len(),
        todo.len()
    );
    let base_dom = canon.root.dom.clone();
    let next = Arc::new(AtomicU64::new(0));
    let done_count = Arc::new(AtomicU64::new(0));
    let done = Arc::new(AtomicBool::new(false));
    let total = todo.len() as u64;
    let live = Arc::new(LiveStats::new(
        total,
        threads.max(1).min(todo.len().max(1)),
        m,
    ));
    // Per-process status file: multi-KB lines are not atomic-append-safe
    // when several processes share one file on NFS (two local runs, or a
    // restarted pod reusing the name), so each process gets its own.
    let status_path = format!("{dir}/status_{}.{}.jsonl", live.pod, std::process::id());
    println!("status: {status_path}");
    let hb = spawn_heartbeat(
        Arc::clone(&live),
        Some(status_path),
        Arc::clone(&done),
    );
    std::thread::scope(|scope| {
        for wid in 0..threads.max(1).min(todo.len().max(1)) {
            let sh = Arc::clone(&sh);
            let next = Arc::clone(&next);
            let done_count = Arc::clone(&done_count);
            let live = Arc::clone(&live);
            let base_dom = &base_dom;
            let todo = &todo;
            let pod = &pod;
            scope.spawn(move || loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i as usize >= todo.len() {
                    break;
                }
                let x0 = todo[i as usize];
                let (st, _samples, _tc, hist) = run_worker(
                    Arc::clone(&sh),
                    base_dom,
                    x0,
                    None,
                    Some(Arc::clone(&live)),
                    wid,
                );
                let json = shard_json(side, m, &st, &hist);
                let path = shard_path(dir, x0 as u32);
                // Fleet-safe durable install (see install_shard). Completion
                // is counted only after the shard is durable: item_done moves
                // the heartbeat's items counter, so it must follow the
                // install, not the search. A failed install leaves the item
                // uncounted; resume picks it up on the next run.
                let tmp = format!("{path}.tmp.{pod}.{}", std::process::id());
                if let Err(e) = std::fs::write(&tmp, &json) {
                    eprintln!("shard {x0:04}: cannot write {tmp}: {e}; item left for resume");
                    live.item_abandon(wid);
                    continue;
                }
                if !install_shard(&tmp, &path, side, m, x0 as u64, &sh) {
                    eprintln!("shard {x0:04}: install failed; item left for resume");
                    live.item_abandon(wid);
                    continue;
                }
                live.item_done(wid);
                let d = done_count.fetch_add(1, Ordering::Relaxed) + 1;
                eprintln!(
                    "shard {:04} done ({}/{}): count={} nodes={} {:.1}s elapsed={:.0}s",
                    x0,
                    d,
                    total,
                    st.count,
                    st.nodes,
                    st.seconds,
                    t0.elapsed().as_secs_f64()
                );
            });
        }
    });
    done.store(true, Ordering::Relaxed);
    // Final synchronous record (see solve): guarantees every run, however
    // short, ends with its terminal counters in the per-process status file.
    if let Some(h) = &hb {
        h.emit();
    }
    // done_count, not total: only verified durable installs are "written";
    // items whose install failed were abandoned above and left for resume,
    // and a fleet incident must never read as a fully written range.
    println!(
        "SHARDS DONE: {}/{} shards written in {:.1}s",
        done_count.load(Ordering::Relaxed),
        total,
        t0.elapsed().as_secs_f64()
    );
    true
}

/// Status-side shard check: the same decoded validation that resume and
/// merge enforce, with side/blocks taken from the manifest when present and
/// from the shard itself otherwise. The validation context (coordinate
/// change plus slot-0 alphabet) per (side, blocks) is cached by the caller;
/// building it skips the expensive pair tables.
fn status_shard_ok(
    v: &serde_json::Value,
    man_side: Option<&str>,
    man_blocks: Option<u64>,
    man_items: Option<&std::collections::HashSet<u64>>,
    ctx_cache: &mut std::collections::HashMap<(u8, u64), (Coords, Vec<K2>)>,
) -> bool {
    let Some(side_s) = v["side"].as_str() else {
        return false;
    };
    if let Some(s) = man_side {
        if side_s != s {
            return false;
        }
    }
    let (side, key) = match side_s {
        "L" => (Side::L, b'L'),
        "R" => (Side::R, b'R'),
        _ => return false,
    };
    if let Some(b) = man_blocks {
        if v["blocks"].as_u64() != Some(b) {
            return false;
        }
    }
    let Some(m64) = man_blocks.or_else(|| v["blocks"].as_u64()) else {
        return false;
    };
    if !(1..=3).contains(&m64) {
        return false;
    }
    let Some(item) = v["item"].as_u64() else {
        return false;
    };
    if let Some(items) = man_items {
        if !items.contains(&item) {
            return false;
        }
    }
    let ctx = ctx_cache
        .entry((key, m64))
        .or_insert_with(|| shard_ctx(side, m64 as usize));
    shard_decode_value(v, side, m64 as usize, item, &ctx.0, &ctx.1).is_some()
}

/// Fleet/local dashboard over a shard directory: items.json manifest,
/// shard_*.json results, and status_<pod>.json[l] heartbeats (one file per
/// pod on a shared volume). Read-only; safe to run mid-flight from any
/// machine that can see the directory.
pub fn run_status(dir: &str) -> bool {
    use std::io::{Read as _, Seek as _, SeekFrom};
    println!("cls-g-csp-status: dir={dir}");
    if std::fs::read_dir(dir).is_err() {
        eprintln!("cannot read directory {dir}");
        return false;
    }
    // Manifest: identity of the run plus the expected item set, used to
    // validate every shard before it enters the aggregate. A manifest that
    // is present but malformed (wrong engine, missing side/blocks,
    // non-numeric or duplicate items) poisons the whole directory: every
    // shard is rejected rather than risk a corrupt 100%-coverage report.
    let mut expected: Option<u64> = None;
    let mut man_side: Option<String> = None;
    let mut man_blocks: Option<u64> = None;
    let mut man_items: Option<std::collections::HashSet<u64>> = None;
    let mut manifest_bad = false;
    match std::fs::read_to_string(format!("{dir}/items.json")) {
        Ok(text) => match serde_json::from_str::<serde_json::Value>(&text) {
            Ok(v) => {
                let items_raw = v["items"].as_array();
                let items_numeric =
                    items_raw.is_some_and(|a| a.iter().all(|x| x.as_u64().is_some()));
                let set: std::collections::HashSet<u64> = items_raw
                    .map(|a| a.iter().filter_map(|x| x.as_u64()).collect())
                    .unwrap_or_default();
                let dupes = items_raw.is_some_and(|a| a.len() != set.len());
                if v["engine"].as_str() == Some(ENGINE_VERSION)
                    && v["side"].as_str().is_some()
                    && v["blocks"].as_u64().is_some()
                    && items_numeric
                    && !dupes
                {
                    expected = Some(set.len() as u64);
                    man_side = v["side"].as_str().map(|s| s.to_string());
                    man_blocks = v["blocks"].as_u64();
                    man_items = Some(set);
                    println!(
                        "manifest: engine={} side={} blocks={} items={}",
                        v["engine"].as_str().unwrap_or("?"),
                        v["side"].as_str().unwrap_or("?"),
                        v["blocks"]
                            .as_u64()
                            .map(|b| b.to_string())
                            .unwrap_or_else(|| "?".into()),
                        expected.map(|e| e.to_string()).unwrap_or_else(|| "?".into()),
                    );
                } else {
                    manifest_bad = true;
                    println!(
                        "items.json: MALFORMED (engine/side/blocks/items); refusing to aggregate shards"
                    );
                }
            }
            Err(_) => {
                manifest_bad = true;
                println!("items.json: unparseable; refusing to aggregate shards");
            }
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            println!("items.json: absent (aggregates will be unverified)")
        }
        Err(e) => {
            // Present but unreadable is not "absent": poison the directory
            // rather than aggregate without manifest verification.
            manifest_bad = true;
            println!("items.json: unreadable ({e}); refusing to aggregate shards");
        }
    }
    // Coverage check: the manifest item set must be exactly the canonical
    // work-item set for its (side, blocks). A shrunken manifest must never
    // produce a 100%-coverage report, and an unknown side/blocks value
    // poisons the directory just like any other malformation.
    if !manifest_bad {
        if let (Some(s), Some(b), Some(items)) =
            (man_side.as_deref(), man_blocks, man_items.as_ref())
        {
            let side = match s {
                "L" => Some(Side::L),
                "R" => Some(Side::R),
                _ => None,
            };
            match (side, b) {
                (Some(sd), 1..=3) => {
                    let canon: std::collections::HashSet<u64> = canonical_run(sd, b as usize)
                        .items
                        .iter()
                        .map(|&x| x as u64)
                        .collect();
                    if &canon != items {
                        manifest_bad = true;
                        println!(
                            "items.json: item set != canonical work-item set for side={s} blocks={b}; refusing to aggregate shards"
                        );
                    }
                }
                _ => {
                    manifest_bad = true;
                    println!(
                        "items.json: MALFORMED (unknown side or blocks out of range); refusing to aggregate shards"
                    );
                }
            }
        }
    }
    let mut ctx_cache: std::collections::HashMap<(u8, u64), (Coords, Vec<K2>)> =
        std::collections::HashMap::new();
    // A shard enters the aggregate only if it passes the exact same decoded
    // validation that resume and merge enforce (status_shard_ok below).
    // Anything else is reported bad, never silently summed.
    let mut n_shards = 0u64;
    let (mut count, mut nodes) = (0u64, 0u64);
    let mut checksum = 0u64;
    let mut shard_secs = 0f64;
    let mut bad = 0u64;
    let mut slowest: Vec<(u64, f64)> = Vec::new();
    let mut seen_items: std::collections::HashSet<u64> = std::collections::HashSet::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            // Canonical shard names only: shard_{item:04}.json, and the
            // embedded item must match the filename. A copied or renamed
            // shard must never be double-counted into the aggregate.
            let Some(stem) = name.strip_prefix("shard_").and_then(|s| s.strip_suffix(".json"))
            else {
                continue;
            };
            let Some(name_item) = stem.parse::<u64>().ok().filter(|_| {
                !stem.is_empty() && stem.chars().all(|c| c.is_ascii_digit())
            }) else {
                continue;
            };
            if name != format!("shard_{name_item:04}.json") {
                bad += 1;
                continue;
            }
            if manifest_bad || !seen_items.insert(name_item) {
                bad += 1;
                continue;
            }
            let v = std::fs::read_to_string(e.path())
                .ok()
                .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok());
            match v {
                Some(v)
                    if v["item"].as_u64() == Some(name_item)
                        && status_shard_ok(
                            &v,
                            man_side.as_deref(),
                            man_blocks,
                            man_items.as_ref(),
                            &mut ctx_cache,
                        ) =>
                {
                    n_shards += 1;
                    count = count.saturating_add(v["count"].as_u64().unwrap_or(0));
                    nodes = nodes.saturating_add(v["nodes"].as_u64().unwrap_or(0));
                    let t = shard_secs + v["seconds"].as_f64().unwrap_or(0.0);
                    shard_secs = if t.is_finite() { t } else { f64::MAX };
                    // Checksum wraps by construction (splitmix64 sum), same
                    // as solve/merge; count/nodes saturate instead.
                    checksum = checksum.wrapping_add(
                        u64::from_str_radix(v["checksum"].as_str().unwrap_or("0"), 16)
                            .unwrap_or(0),
                    );
                    slowest.push((name_item, v["seconds"].as_f64().unwrap_or(0.0)));
                }
                _ => bad += 1,
            }
        }
    }
    slowest.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    slowest.truncate(5);
    println!(
        "shards: {} complete ({} rejected){}; solutions={} nodes={} core_secs={:.0} checksum={:016x}",
        n_shards,
        bad,
        if expected.is_some() || manifest_bad {
            ""
        } else {
            " [no manifest; identity unverified]"
        },
        count,
        nodes,
        shard_secs,
        checksum
    );
    // Coverage only means something against a manifest that passed the
    // canonical-set check; a poisoned directory reports rejections above,
    // not a percentage. exp > 0 also keeps a zero-item manifest off NaN.
    if let Some(exp) = expected.filter(|&e| !manifest_bad && e > 0) {
        println!(
            "coverage: {}/{} items done ({:.1}%), {} remaining",
            n_shards,
            exp,
            100.0 * n_shards as f64 / exp as f64,
            exp.saturating_sub(n_shards)
        );
    }
    if !slowest.is_empty() {
        println!(
            "slowest items: {}",
            slowest
                .iter()
                .map(|(i, s)| format!("{i}:{s:.0}s"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    // Heartbeats: the atomic snapshot (status_<pod>.json) is always complete
    // by construction. Fallback: the newest parseable line in the last 1MB
    // of the jsonl history (never a torn mid-append tail, never a full-file
    // read of an unbounded log). Both paths pass the same schema check.
    let hb_ok = |v: &serde_json::Value| {
        v["pod"].is_string()
            && v["nodes"].is_u64()
            && v["rate_avg"].is_f64()
            && v["items_done"].is_u64()
            && v["items_total"].is_u64()
    };
    let read_pod = |stem: &str| -> Option<(serde_json::Value, String)> {
        let snap = format!("{dir}/status_{stem}.json");
        if let Ok(text) = std::fs::read_to_string(&snap) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
                if hb_ok(&v) {
                    return Some((v, snap));
                }
            }
        }
        let path = format!("{dir}/status_{stem}.jsonl");
        let mut f = std::fs::File::open(&path).ok()?;
        let len = f.metadata().ok()?.len();
        let _ = f.seek(SeekFrom::Start(len.saturating_sub(1 << 20)));
        let mut buf = String::new();
        let _ = f.read_to_string(&mut buf);
        for line in buf.lines().rev() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                if hb_ok(&v) {
                    return Some((v, path));
                }
            }
        }
        None
    };
    let mut stems: Vec<String> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            for suffix in [".jsonl", ".json"] {
                if let Some(stem) = name
                    .strip_prefix("status_")
                    .and_then(|s| s.strip_suffix(suffix))
                {
                    if !name.ends_with(".tmp") && !stems.iter().any(|s| s == stem) {
                        stems.push(stem.to_string());
                    }
                }
            }
        }
    }
    stems.sort();
    let mut any_hb = false;
    for stem in stems {
        let Some((v, path)) = read_pod(&stem) else {
            continue;
        };
        any_hb = true;
        let age = std::fs::metadata(&path)
            .and_then(|md| md.modified())
            .ok()
            .and_then(|t| std::time::SystemTime::now().duration_since(t).ok())
            .map(|d| d.as_secs());
        let workers: Vec<String> = v["workers"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|w| {
                        format!(
                            "w{}#{}n={}d={}t={}s{}",
                            w["wid"].as_u64().unwrap_or(0),
                            w["item"].as_u64().unwrap_or(0),
                            w["nodes"].as_u64().unwrap_or(0),
                            w["depth"].as_u64().unwrap_or(0),
                            w["secs_on_item"].as_u64().unwrap_or(0),
                            w["ref_pct_of_expected"]
                                .as_f64()
                                .map(|p| format!("({p:.0}%)"))
                                .unwrap_or_default(),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        println!(
            "pod {}: file_age={}s nodes={} rate={:.0}/s sols={} items={}/{} eta={} rss={:.0}MB",
            v["pod"].as_str().unwrap_or("?"),
            age.map(|a| a.to_string()).unwrap_or_else(|| "?".into()),
            v["nodes"].as_u64().unwrap_or(0),
            v["rate_avg"].as_f64().unwrap_or(0.0),
            v["solutions"].as_u64().unwrap_or(0),
            v["items_done"].as_u64().unwrap_or(0),
            v["items_total"].as_u64().unwrap_or(0),
            v["eta_s"]
                .as_f64()
                .map(|e| format!("{:.1}h", e / 3600.0))
                .unwrap_or_else(|| "?".into()),
            v["rss_mb"].as_f64().unwrap_or(0.0),
        );
        if !workers.is_empty() {
            println!("  workers: {}", workers.join(" "));
        }
    }
    if !any_hb {
        println!("status_<pod>.json[l]: none found (heartbeats off or no run in flight)");
    }
    !manifest_bad && (expected.is_some() || n_shards > 0 || any_hb)
}

/// Merge a shard directory into a single census artifact. Verifies complete
/// prefix coverage and per-shard integrity; refuses (returns false) on any
/// missing, unreadable, or incompatible shard. Merge is commutative.
pub fn run_merge(side: Side, m: usize, dir: &str, out_path: &str) -> bool {
    println!(
        "cls-g-csp-merge: side={} blocks={} dir={}",
        side.name(),
        m,
        dir
    );
    sweep_stale_tmps(dir);
    let Ok(items_text) = std::fs::read_to_string(format!("{dir}/items.json")) else {
        eprintln!("missing items.json in {dir}; run cls-g-csp-shard first");
        return false;
    };
    let Ok(items_v) = serde_json::from_str::<serde_json::Value>(&items_text) else {
        eprintln!("items.json is not valid JSON");
        return false;
    };
    if items_v["engine"].as_str() != Some(ENGINE_VERSION)
        || items_v["side"].as_str() != Some(side.name())
        || items_v["blocks"].as_u64() != Some(m as u64)
    {
        eprintln!("items.json is incompatible with this merge (engine/side/blocks)");
        return false;
    }
    let items: Vec<u32> = match items_v["items"].as_array() {
        Some(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
            let mut bad = false;
            for v in arr {
                match v.as_u64() {
                    // A duplicated item would double-count its shard while
                    // the class-consistency check still passed (both totals
                    // double together), so duplicates are fatal here.
                    Some(x) if x <= u32::MAX as u64 && seen.insert(x as u32) => {
                        out.push(x as u32)
                    }
                    _ => {
                        bad = true;
                        break;
                    }
                }
            }
            if bad {
                eprintln!("items.json: items array contains non-numeric, out-of-range, or duplicate values");
                return false;
            }
            out
        }
        None => {
            eprintln!("items.json: missing items array");
            return false;
        }
    };
    // Coverage definition: the manifest item set must be exactly the
    // canonical work-item set for (side, m), recomputed here. A shrunken
    // manifest must never yield a "complete" artifact.
    let canon = canonical_run(side, m);
    let canon_set: std::collections::HashSet<u64> =
        canon.items.iter().map(|&x| x as u64).collect();
    let man_set: std::collections::HashSet<u64> = items.iter().map(|&x| x as u64).collect();
    if canon_set != man_set {
        eprintln!(
            "items.json item set does not match the canonical work-item set for side={} blocks={}; refusing to merge",
            side.name(),
            m
        );
        return false;
    }
    let mut ok = true;
    let (mut count, mut checksum, mut nodes) = (0u64, 0u64, 0u64);
    let mut seconds = 0f64;
    let mut hist: HashMap<ClassKey, ClassRec> = HashMap::new();
    let mut items_detail: Vec<ItemStat> = Vec::new();
    for &item in &items {
        let path = shard_path(dir, item);
        let Some(sd) = shard_decode(&path, side, m, item as u64, &canon.sh.coords, &canon.sh.slots[0])
        else {
            eprintln!("MISSING or INVALID shard for item {item}: {path}");
            ok = false;
            continue;
        };
        // Aggregate first, with checks, before any histogram mutation: the
        // per-shard class counts already sum (checked, in shard_decode) to
        // sd.count, so once the running total passes checked_add the
        // histogram additions are bounded by that total and cannot
        // overflow.
        let (Some(nc), Some(nn)) = (count.checked_add(sd.count), nodes.checked_add(sd.nodes))
        else {
            eprintln!("aggregate count/nodes overflow at item {item}");
            return false;
        };
        let ns = seconds + sd.seconds;
        if !ns.is_finite() {
            eprintln!("aggregate seconds overflow at item {item}");
            return false;
        }
        count = nc;
        nodes = nn;
        seconds = ns;
        checksum = checksum.wrapping_add(sd.checksum);
        for (k, rec) in sd.classes {
            merge_hist(&mut hist, HashMap::from([(k, rec)]));
        }
        items_detail.push(ItemStat {
            item,
            count: sd.count,
            checksum: sd.checksum,
            nodes: sd.nodes,
            seconds: sd.seconds,
        });
    }
    if !ok {
        eprintln!("merge REFUSED: census incomplete or inconsistent");
        return false;
    }
    // Class-consistency: merged histogram must reproduce the total.
    let total_from_classes: u64 = hist.values().map(|r| r.count).sum();
    if total_from_classes != count {
        eprintln!("merged class total {total_from_classes} != count {count}");
        return false;
    }
    items_detail.sort_by_key(|st| st.item);
    let items_json: Vec<String> = items_detail
        .iter()
        .map(|st| {
            format!(
                "{{\"item\":{},\"count\":{},\"checksum\":\"{:016x}\",\"nodes\":{},\"seconds\":{:.3}}}",
                st.item, st.count, st.checksum, st.nodes, st.seconds
            )
        })
        .collect();
    let mut by_count: Vec<(&ClassKey, &ClassRec)> = hist.iter().collect();
    // Same total order as classes_to_json: the samples selection must be as
    // deterministic as the classes array or merged artifacts differ by run.
    by_count.sort_by(class_order);
    let sample_json: Vec<String> = by_count.iter().take(8).map(|(_, r)| mat_json(&r.rep)).collect();
    let json = format!(
        "{{\n  \"source\": \"arXiv:2408.09342 Eq 8.2, CLS Appendix C basis; full enumeration via K=Q(sqrt(-3)) commutant reduction, v2 slot-level CSP engine (src/four_color/gmatrix_csp.rs), merged from durable shards\",\n  \"engine\": \"{}\",\n  \"side\": \"{}\",\n  \"blocks\": {},\n  \"dim\": {},\n  \"count\": {},\n  \"checksum_splitmix64_sum\": \"{:016x}\",\n  \"nodes\": {},\n  \"shard_seconds\": {:.3},\n  \"complete\": true,\n  \"work_items\": {},\n  \"note\": \"Merged from {} immutable per-prefix shards after full coverage verification; every class representative independently re-verified against G^2 = A at merge time. Checksum identical in construction to gmatrix_full.\",\n  \"items_detail\": [{}],\n  \"classes\": {},\n  \"samples\": [{}]\n}}\n",
        ENGINE_VERSION,
        side.name(),
        m,
        4 * m,
        count,
        checksum,
        nodes,
        seconds,
        items.len(),
        items.len(),
        items_json.join(","),
        classes_to_json(&hist, m),
        sample_json.join(",")
    );
    let tmp = format!("{out_path}.tmp.{}.{}", pod_name(), std::process::id());
    if let Err(e) = std::fs::write(&tmp, &json) {
        eprintln!("cannot write {tmp}: {e}");
        return false;
    }
    if let Err(e) = std::fs::rename(&tmp, out_path) {
        eprintln!("cannot rename {tmp} to {out_path}: {e}");
        let _ = std::fs::remove_file(&tmp);
        return false;
    }
    println!(
        "MERGED: count={} checksum={:016x} nodes={} shards={} -> {out_path}",
        count,
        checksum,
        nodes,
        items.len()
    );
    true
}

// ===========================================================================
// CLI entry: build + artifact (same JSON schema as gmatrix_full)
// ===========================================================================

pub fn run_build(side: Side, m: usize, threads: usize, cap: Option<u64>, stride: usize, path: &str) {
    println!(
        "cls-g-csp: side={} blocks={} threads={} cap={:?} stride={}",
        side.name(),
        m,
        threads,
        cap,
        stride
    );
    let coords = build_coords(m, side);
    let (_, blocks) = cls_a_blocks(m, side);
    let alph = build_alphabets(&coords, &blocks);
    println!("intertwiner counts: {:?}", alph.intertwiner_counts);

    // Same sanity ladder as v1: per-slot M_2(K) counts and 12^m block diagonal.
    for s in 0..m {
        let sols = slot_solutions(&coords, &alph, s);
        let expect = gmatrix::g_matrices(&blocks[s]).len();
        println!("slot {s}: M_2(K) solutions = {} (gmatrix per-block = {expect})", sols.len());
        assert_eq!(sols.len(), expect, "slot {s} M_2(K) count mismatch");
    }
    let bd = block_diagonal_via_coords(&coords, &alph);
    println!("block-diagonal via coords = {bd} (expect 12^{m})");
    assert_eq!(bd, 12usize.pow(m as u32));

    let sh = Arc::new(build_shared(coords, &alph));
    let stats = solve(sh, threads, cap, stride);
    println!(
        "DONE: count={} checksum={:016x} nodes={} seconds={:.1} complete={} tight_boxes={}",
        stats.count, stats.checksum, stats.nodes, stats.seconds, stats.complete, stats.tight_computes
    );

    let sample_json: Vec<String> = stats
        .samples
        .iter()
        .map(|g| {
            let rows: Vec<String> = g
                .iter()
                .map(|r| format!("[{}]", r.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",")))
                .collect();
            format!("[{}]", rows.join(","))
        })
        .collect();
    let items_json: Vec<String> = stats
        .items_detail
        .iter()
        .map(|st| {
            format!(
                "{{\"item\":{},\"count\":{},\"checksum\":\"{:016x}\",\"nodes\":{},\"seconds\":{:.3}}}",
                st.item, st.count, st.checksum, st.nodes, st.seconds
            )
        })
        .collect();
    let classes_json = classes_to_json(&stats.hist, m);
    let json = format!(
        "{{\n  \"source\": \"arXiv:2408.09342 Eq 8.2, CLS Appendix C basis; full enumeration via K=Q(sqrt(-3)) commutant reduction, v2 slot-level CSP engine (src/four_color/gmatrix_csp.rs)\",\n  \"engine\": \"gmatrix_csp\",\n  \"side\": \"{}\",\n  \"blocks\": {},\n  \"dim\": {},\n  \"count\": {},\n  \"checksum_splitmix64_sum\": \"{:016x}\",\n  \"nodes\": {},\n  \"seconds\": {:.3},\n  \"complete\": {},\n  \"stride\": {},\n  \"work_items\": {},\n  \"shared_product_table\": {},\n  \"table_bytes\": {},\n  \"tight_box_computes\": {},\n  \"note\": \"All G in {{-1,0,1}}^(4m x 4m) with G^2 = A. Slot-level CSP: correlated product boxes (outward-widened f64), exact residual arc-consistency, MRV order. Every counted solution mapped back to an integer G and verified G^2 = A exactly. Checksum identical in construction to gmatrix_full. items_detail is the per-prefix immutable record (item, count, checksum, nodes, seconds); totals are commutative sums over it.\",\n  \"items_detail\": [{}],\n  \"classes\": {},\n  \"samples\": [{}]\n}}\n",
        side.name(),
        m,
        4 * m,
        stats.count,
        stats.checksum,
        stats.nodes,
        stats.seconds,
        stats.complete,
        stride,
        stats.items,
        stats.shared_table,
        stats.table_bytes,
        stats.tight_computes,
        items_json.join(","),
        classes_json,
        sample_json.join(",")
    );
    std::fs::create_dir_all("results").ok();
    // Pod/pid-unique tmp: concurrent builds to the same artifact path must
    // never interleave tmp writes. Build artifacts live outside the swept
    // shard directories, so a crash-leaked tmp here is simply inert.
    let tmp = format!("{path}.tmp.{}.{}", pod_name(), std::process::id());
    if let Err(e) = std::fs::write(&tmp, &json).and_then(|()| std::fs::rename(&tmp, path)) {
        eprintln!("cannot write artifact {path}: {e}");
        let _ = std::fs::remove_file(&tmp);
        return;
    }
    println!("wrote {path}");
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn setup(m: usize) -> (Coords, Alphabets) {
        let coords = build_coords(m, Side::L);
        let (_, blocks) = cls_a_blocks(m, Side::L);
        let alph = build_alphabets(&coords, &blocks);
        (coords, alph)
    }

    #[test]
    fn slot_lists_documented_sharing_property() {
        // If the three CLS blocks are equal, all nine slot lists are equal and
        // one product table serves all pairs. Either way the engine is correct;
        // this documents which path is active.
        let (_, alph) = setup(3);
        let identical = alph.slots.iter().all(|s| *s == alph.slots[0]);
        println!("slot lists identical across slots: {identical}");
        assert!(alph.slots.iter().all(|s| s.len() == alph.slots[0].len()));
    }

    #[test]
    fn csp_m1_count_is_12() {
        for threads in [1, 3] {
            let (coords, alph) = setup(1);
            let sh = Arc::new(build_shared(coords, &alph));
            let st = solve(sh, threads, None, 1);
            assert_eq!(st.count, 12, "m=1 count must be 12 (threads {threads})");
            assert!(st.complete);
        }
    }

    #[test]
    fn csp_m1_checksum_matches_v1() {
        let (coords, alph) = setup(1);
        let sh = Arc::new(build_shared(coords, &alph));
        let st = solve(sh, 1, None, 1);
        assert_eq!(
            format!("{:016x}", st.checksum),
            "4b3aa9ef562965aa",
            "m=1 checksum must match the v1 artifact"
        );
    }

    /// Full m=2 cross-check against the v1 census: count 15000 and checksum
    /// 94e85bc1c8e786fd, the strongest pre-m=3 validation available. The node
    /// count is also locked: the search tree shape is part of the validated
    /// object, so any optimization or instrumentation that perturbs pruning
    /// decisions fails here even when the solutions stay correct.
    #[test]
    fn csp_m2_matches_v1_census() {
        let (coords, alph) = setup(2);
        let sh = Arc::new(build_shared(coords, &alph));
        let st = solve(sh, 4, None, 1);
        assert_eq!(st.count, 15000, "m=2 count must match v1");
        assert_eq!(
            format!("{:016x}", st.checksum),
            "94e85bc1c8e786fd",
            "m=2 checksum must match v1"
        );
        assert_eq!(st.nodes, 15211, "m=2 node count must match the validated tree");
    }

    /// Round-6 regression for the copied-shard attack: a valid shard whose
    /// item field is rewritten to a different work item must be refused,
    /// because every class rep's leading 4x4 block is bound to the claimed
    /// item's slot-0 value. Also: out-of-range item indices refuse cleanly.
    #[test]
    fn csp_shard_decode_binds_item() {
        let canon = canonical_run(Side::L, 1);
        let sh = &canon.sh;
        let x0 = canon.items[0];
        let (st, _samples, _tc, hist) =
            run_worker(Arc::clone(sh), &canon.root.dom, x0, None, None, 0);
        let json = shard_json(Side::L, 1, &st, &hist);
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(
            shard_decode_value(&v, Side::L, 1, x0 as u64, &sh.coords, &sh.slots[0]).is_some(),
            "an honest shard must decode"
        );
        let other = canon.items[1];
        let mut forged = v.clone();
        forged["item"] = serde_json::json!(other);
        assert!(
            shard_decode_value(&forged, Side::L, 1, other as u64, &sh.coords, &sh.slots[0])
                .is_none(),
            "the same bytes under a rewritten item field must be refused"
        );
        assert!(
            shard_decode_value(
                &v,
                Side::L,
                1,
                sh.slots[0].len() as u64,
                &sh.coords,
                &sh.slots[0]
            )
            .is_none(),
            "an out-of-range item index must refuse, not panic"
        );
    }

    /// One-off measurement: distinct exact products per term pair, and
    /// singularity stats. Decides the memory feasibility of the
    /// product-to-support lookup maps. Run with --ignored --nocapture.
    #[test]
    #[ignore]
    fn measure_distinct_products() {
        let (_, alph) = setup(3);
        let m = 3;
        let mut grand = 0usize;
        for bi in 0..m {
            for bj in 0..m {
                for bk in 0..m {
                    let (l, r) = (bi * m + bk, bk * m + bj);
                    let a = &alph.slots[l];
                    let b = &alph.slots[r];
                    let mut set = std::collections::HashSet::new();
                    if l == r {
                        for x in a {
                            set.insert(mul22(x, x));
                        }
                    } else {
                        for x in a {
                            for y in b {
                                set.insert(mul22(x, y));
                            }
                        }
                    }
                    grand += set.len();
                    let sing = a.iter().filter(|x| inv22(x).is_none()).count();
                    println!(
                        "term ({bi},{bj},{bk}) pair ({l},{r}): distinct products = {}, left singular {sing}/{}",
                        set.len(),
                        a.len()
                    );
                }
            }
        }
        println!("TOTAL distinct products across 27 pairs: {grand}");
        println!("approx map memory: {} MB", grand * 400 / 1_000_000);
    }
}
