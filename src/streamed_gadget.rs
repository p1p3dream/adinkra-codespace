//! Memory-bounded gadget via the Gram-matrix identity.
//!
//! The dense path (`decompose::dense_gadget_matrix`) retains one `DenseHoloraumy`
//! per irreducible summand simultaneously (C(N,2) matrices of dmin x dmin f64,
//! ~15.7 MB at N=16). That scales with `num_irreps` and OOM-kills at k=7 (~36 GB).
//!
//! Key identity that fixes it: each fermionic holoraumy `Vtilde_IJ` (I != J) is
//! ANTISYMMETRIC (it is the I != J Garden relation `L_I R_J + L_J R_I = 0`, i.e.
//! `Vtilde_IJ^T = -Vtilde_IJ`). For an antisymmetric `B`,
//!   Tr(A . B) = sum_{a,b} A[a][b] B[b][a] = -sum_{a,b} A[a][b] B[a][b] = -<A,B>_F.
//! Hence
//!   G[i][j] = -2/(N(N-1) dmin) sum_{I>J} Tr(Vtilde_i_IJ . Vtilde_j_IJ)
//!           = +2/(N(N-1) dmin) sum_{I>J} <Vtilde_i_IJ, Vtilde_j_IJ>_F
//!           = c * <w_i, w_j>,   c = 2/(N(N-1) dmin),
//! where `w_i` is the flat concatenation of all C(N,2) `Vtilde` matrices of
//! summand i. So the whole gadget matrix is `c * Gram(W)`.
//!
//! Self-check: <w_i,w_i> = sum_IJ ||Vtilde_IJ||_F^2 = sum_IJ (-Tr(Vtilde^2)) =
//! C(N,2)*dmin, so G[i][i] = c*C(N,2)*dmin = 1 (matches the irreducible self-gadget).
//!
//! Implementation: store each `w_i` as f32 (half the RAM of the f64 holoraumy
//! tensors) and accumulate the dot products in f64. The symmetric Gram is computed
//! in parallel over rows (upper triangle, then mirrored). This is BLAS-shaped:
//! the inner loop is a plain dot product over contiguous vectors, so a later
//! swap to a tiled GEMM / cuBLAS backend is a block-multiply change, not a math
//! rewrite. Peak extra memory is the flat vectors (`num_irreps * C(N,2) * dmin^2 *
//! 4` bytes) plus the result matrix, NOT the f64 holoraumy tensors.

#![allow(dead_code)] // primitive-library module: much of its API surface is exercised by the test suite, not the binary main path

use crate::decompose::{DenseHoloraumy, IrrepSummand};
use crate::holoraumy::dmin;

/// Length of a flattened holoraumy vector: C(n,2) * dmin(n)^2.
pub fn flat_len(n: usize) -> usize {
    (n * (n - 1) / 2) * dmin(n) * dmin(n)
}

/// f32 bytes of the full flat-vector store for `num_irreps` summands at `n`.
pub fn flat_store_bytes(n: usize, num_irreps: usize) -> u64 {
    (num_irreps as u64) * (flat_len(n) as u64) * 4
}

/// f64 bytes of the full flat-vector store (the `--f64` exact disk build, double
/// the f32 store: ~145 GiB for the k=5 stratum at N=16).
pub fn flat_store_bytes_f64(n: usize, num_irreps: usize) -> u64 {
    (num_irreps as u64) * (flat_len(n) as u64) * 8
}

/// Flatten a summand's fermionic holoraumy (all C(n,2) `Vtilde` matrices,
/// row-major, in the same I>J pair order as `DenseHoloraumy::from_summand`) into
/// one contiguous f32 vector. The transient `DenseHoloraumy` is dropped on return,
/// so only the f32 vector is retained.
pub fn flatten_summand(s: &IrrepSummand) -> Vec<f32> {
    let dh = DenseHoloraumy::from_summand(s);
    let mut w = Vec::with_capacity(dh.vtilde.iter().map(|m| m.data.len()).sum());
    for m in &dh.vtilde {
        for &x in &m.data {
            w.push(x as f32);
        }
    }
    w
}

/// Gadget value between two flat holoraumy vectors: `c * <a,b>` with f64
/// accumulation.
///
/// Equal to `decompose::dense_gadget` on the corresponding summands UP TO f32
/// storage quantization: the underlying identity is exact, but `w` is stored in
/// f32, so the result is exact only when the holoraumy entries are exactly
/// f32-representable (e.g. the 0, ±1 signed-permutation entries of the k=8
/// stratum, where it reproduces the dense path bit-for-bit) and carries ~1e-7
/// relative error for the dense restricted operators of reducible strata (k<8).
/// The gadget VALUES are therefore good to ~1e-7; do NOT treat a count of
/// "distinct off-diagonal values" as exact (f32 quantization can split/merge
/// nearby values — and for k<8 those values are orientation-dependent anyway, so
/// any such count is a run-specific fingerprint, not a classification).
#[inline]
pub fn gadget_from_flat(a: &[f32], b: &[f32], n: usize) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f64;
    for k in 0..a.len() {
        acc += a[k] as f64 * b[k] as f64;
    }
    let dmin_val = dmin(n);
    2.0 * acc / (n * (n - 1) * dmin_val) as f64
}

/// Flatten a summand's fermionic holoraumy directly into a preallocated f32 slice
/// of length `flat_len(n)` (avoids a per-summand `Vec` allocation, so the caller
/// can fill one big contiguous `W` buffer in place — no Vec-of-Vec, no 2x copy).
pub fn flatten_summand_into(s: &IrrepSummand, out: &mut [f32]) {
    let dh = DenseHoloraumy::from_summand(s);
    let mut idx = 0;
    for m in &dh.vtilde {
        for &x in &m.data {
            out[idx] = x as f32;
            idx += 1;
        }
    }
    assert_eq!(idx, out.len(), "flatten_summand_into: length mismatch");
}

/// f64 counterpart of [`flatten_summand_into`]: stores the holoraumy entries as
/// EXACT f64 (no f32 cast), so the disk Gram is exact to ~1e-13 and the distinct
/// off-diagonal value count is trustworthy (the `--f64` build, vs the f32 store
/// whose ~1e-7 quantization can merge/split nearby values).
pub fn flatten_summand_into_f64(s: &IrrepSummand, out: &mut [f64]) {
    let dh = DenseHoloraumy::from_summand(s);
    let mut idx = 0;
    for m in &dh.vtilde {
        for &x in &m.data {
            out[idx] = x;
            idx += 1;
        }
    }
    assert_eq!(idx, out.len(), "flatten_summand_into_f64: length mismatch");
}

/// Gadget matrix from a CONTIGUOUS row-major `W` (`ni x l`, f32) via a
/// cache-blocked GEMM: `G = c * W Wᵀ`, `c = 2/(N(N-1)dmin)`. This replaces the
/// bandwidth-bound per-pair dot loop — the GotoBLAS-style blocking in
/// `matrixmultiply` reuses each tile across many outputs, turning the problem
/// from memory-bound to compute-bound. Accumulation is f32: exact for the 0,±1
/// entries of the k=8 stratum, and ~1e-7 relative for the general-real entries of
/// k<8 (empirically ~1e-16 for the measured k=5/k=7 summands, whose restricted
/// operators turn out near-exactly f32-representable — see `decompose-audit`).
/// Extra memory is the f32 product buffer + the f64 result.
pub fn gram_from_contiguous(w: &[f32], ni: usize, l: usize, n: usize) -> Vec<Vec<f64>> {
    assert_eq!(w.len(), ni * l, "gram_from_contiguous: W shape mismatch");
    let mut prod = vec![0.0f32; ni * ni];
    // C = W * Wᵀ : A = W (ni x l; row stride l, col stride 1);
    //              B = Wᵀ (l x ni; element (p,j) = W[j*l+p] => row stride 1, col stride l).
    if ni > 0 && l > 0 {
        unsafe {
            matrixmultiply::sgemm(
                ni, l, ni,
                1.0,
                w.as_ptr(), l as isize, 1,
                w.as_ptr(), 1, l as isize,
                0.0,
                prod.as_mut_ptr(), ni as isize, 1,
            );
        }
    }
    let scale = 2.0 / (n * (n - 1) * dmin(n)) as f64;
    (0..ni)
        .map(|i| (0..ni).map(|j| prod[i * ni + j] as f64 * scale).collect())
        .collect()
}

/// f64-accumulation variant of [`gram_from_contiguous`] (input `W` in f64). Used
/// by the audit to quantify the f32 error against an exact-in-basis reference.
pub fn gram_from_contiguous_f64(w: &[f64], ni: usize, l: usize, n: usize) -> Vec<Vec<f64>> {
    assert_eq!(w.len(), ni * l);
    let mut prod = vec![0.0f64; ni * ni];
    if ni > 0 && l > 0 {
        unsafe {
            matrixmultiply::dgemm(
                ni, l, ni,
                1.0,
                w.as_ptr(), l as isize, 1,
                w.as_ptr(), 1, l as isize,
                0.0,
                prod.as_mut_ptr(), ni as isize, 1,
            );
        }
    }
    let scale = 2.0 / (n * (n - 1) * dmin(n)) as f64;
    (0..ni)
        .map(|i| (0..ni).map(|j| prod[i * ni + j] * scale).collect())
        .collect()
}

/// Convenience wrapper (tests / small inputs): build a contiguous `W` from a
/// slice of flat f32 vectors and call [`gram_from_contiguous`]. The pipeline
/// builds `W` in place instead (no Vec-of-Vec) to keep peak memory bounded.
pub fn gram_gadget_matrix(vectors: &[Vec<f32>], n: usize) -> Vec<Vec<f64>> {
    let ni = vectors.len();
    if ni == 0 {
        return Vec::new();
    }
    let l = vectors[0].len();
    let mut w = Vec::with_capacity(ni * l);
    for v in vectors {
        w.extend_from_slice(v);
    }
    gram_from_contiguous(&w, ni, l, n)
}

// ===========================================================================
// Disk-backed tiled Gram (for strata whose flat store exceeds RAM, e.g. k=5).
// ===========================================================================
//
// The flat store W (num_irreps x l f32) is written to a scratch file (offset of
// summand g is g*l*4 bytes), then the Gram G = c*W W^T is computed in TILES:
// only `tile_rows` summands of W are resident at once. Math is identical to
// gram_from_contiguous (the contraction axis l is never tiled, so there are no
// cross-tile terms); only the upper triangle is computed and mirrored.

use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

/// Reinterpret an f32 slice as bytes (native-endian; sound — every bit pattern is
/// a valid f32). The scratch file is written and read on the same platform (or
/// little-endian peers: macOS dev + Linux stonkbot are both LE).
fn f32_as_bytes(s: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}
fn f32_as_bytes_mut(s: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut u8, std::mem::size_of_val(s)) }
}
/// f64 byte views (same native-endian soundness as the f32 casts; used by the
/// exact `--f64` disk build).
fn f64_as_bytes(s: &[f64]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}
fn f64_as_bytes_mut(s: &mut [f64]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut u8, std::mem::size_of_val(s)) }
}

/// RAII handle to the on-disk flat-store W. Removes the file on drop unless
/// `ADINKRA_KEEP_SCRATCH=1` (kept for debugging). Positioned reads/writes
/// (`write_at`/`read_exact_at`) use no shared cursor, so disjoint-offset writes
/// from rayon workers are safe without locking.
pub struct ScratchW {
    file: File,
    path: PathBuf,
    keep: bool,
}

impl ScratchW {
    /// Create + preallocate a scratch file of `len_bytes` (sparse via set_len;
    /// every byte is overwritten exactly once by `write_summand`).
    pub fn create(path: PathBuf, len_bytes: u64) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .read(true).write(true).create(true).truncate(true)
            .open(&path)?;
        file.set_len(len_bytes)?;
        let keep = std::env::var("ADINKRA_KEEP_SCRATCH").map(|v| v == "1").unwrap_or(false);
        Ok(ScratchW { file, path, keep })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Positioned byte write with short-write + EINTR handling. The element-typed
    /// `write_summand*` methods delegate here; `write_at` has no shared cursor, so
    /// disjoint-offset writes from rayon workers are safe without locking.
    fn write_bytes_at(&self, mut off: u64, mut buf: &[u8]) -> std::io::Result<()> {
        while !buf.is_empty() {
            match self.file.write_at(buf, off) {
                Ok(0) => return Err(std::io::Error::new(
                    std::io::ErrorKind::WriteZero, "write_at returned 0")),
                Ok(nw) => { buf = &buf[nw..]; off += nw as u64; }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    /// Write summand `g`'s flat vector (length `l`) at byte offset `g*l*4`. Safe
    /// to call concurrently from multiple threads for distinct `g` (disjoint
    /// ranges). Handles short writes.
    pub fn write_summand(&self, g: usize, vec: &[f32], l: usize) -> std::io::Result<()> {
        assert_eq!(vec.len(), l, "write_summand: vec.len() != l (layout corruption)");
        self.write_bytes_at((g as u64) * (l as u64) * 4, f32_as_bytes(vec))
    }

    /// f64 counterpart of [`write_summand`] (byte offset `g*l*8`). Used by the
    /// exact `--f64` disk build.
    pub fn write_summand_f64(&self, g: usize, vec: &[f64], l: usize) -> std::io::Result<()> {
        assert_eq!(vec.len(), l, "write_summand_f64: vec.len() != l (layout corruption)");
        self.write_bytes_at((g as u64) * (l as u64) * 8, f64_as_bytes(vec))
    }

    /// Read `rows` summands starting at summand `row0` into `buf` (len rows*l).
    fn read_rows(&self, row0: usize, rows: usize, l: usize, buf: &mut [f32]) -> std::io::Result<()> {
        assert_eq!(buf.len(), rows * l, "read_rows: buf.len() != rows*l");
        let off = (row0 as u64) * (l as u64) * 4;
        self.file.read_exact_at(f32_as_bytes_mut(buf), off)
    }

    /// f64 counterpart of [`read_rows`] (8-byte stride).
    fn read_rows_f64(&self, row0: usize, rows: usize, l: usize, buf: &mut [f64]) -> std::io::Result<()> {
        assert_eq!(buf.len(), rows * l, "read_rows_f64: buf.len() != rows*l");
        let off = (row0 as u64) * (l as u64) * 8;
        self.file.read_exact_at(f64_as_bytes_mut(buf), off)
    }

    /// Flush to disk (call between the write phase and the read/Gram phase).
    pub fn sync(&self) -> std::io::Result<()> {
        self.file.sync_all()
    }
}

impl Drop for ScratchW {
    fn drop(&mut self) {
        if !self.keep {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Default tile height (summands per W tile). `tile_rows*l*4` bytes per loaded
/// tile; with l≈1.97M f32 at N=16, tile_rows=1024 ≈ 8 GiB/tile (two tiles ≈ 16
/// GiB resident).
pub const DEFAULT_TILE_ROWS: usize = 1024;

/// Tiled Gram `G = c * W Wᵀ` read from a disk scratch (W = `ni x l` f32). Result
/// is the full `ni x ni` f64 matrix in RAM (the result is small relative to W);
/// peak extra memory is two W tiles + the result. Upper triangle computed, then
/// mirrored. The A-block stays resident across the inner loop (read once per
/// outer tile) to minimize disk re-reads.
///
/// Parallelism: the tile loop is sequential, but each `sgemm` is multi-threaded
/// (matrixmultiply `threading` feature), so all cores are used within a tile.
/// Cross-tile parallelism is deferred: the mirror writes `g[gj][gi]` would race on
/// the shared `Vec<Vec<f64>>` rows, so a parallel version needs a flat result with
/// disjoint-slice writes — a future optimization, not a correctness need.
pub fn gram_from_disk(
    scratch: &ScratchW,
    ni: usize,
    l: usize,
    n: usize,
    tile_rows: usize,
) -> std::io::Result<Vec<Vec<f64>>> {
    let scale = 2.0 / (n * (n - 1) * dmin(n)) as f64;
    let mut g = vec![vec![0.0f64; ni]; ni];
    if ni == 0 || l == 0 {
        return Ok(g);
    }
    let t = tile_rows.max(1);
    let num_tiles = ni.div_ceil(t);
    let mut a = vec![0.0f32; t * l];
    let mut b = vec![0.0f32; t * l];
    let mut prod = vec![0.0f32; t * t];

    for bi in 0..num_tiles {
        let ti = t.min(ni - bi * t);
        scratch.read_rows(bi * t, ti, l, &mut a[..ti * l])?;
        for bj in bi..num_tiles {
            let tj = t.min(ni - bj * t);
            let bptr = if bj == bi {
                a.as_ptr()
            } else {
                scratch.read_rows(bj * t, tj, l, &mut b[..tj * l])?;
                b.as_ptr()
            };
            // prod (ti x tj) = A(ti x l) * Bᵀ(l x tj), same strides as gram_from_contiguous.
            unsafe {
                matrixmultiply::sgemm(
                    ti, l, tj,
                    1.0,
                    a.as_ptr(), l as isize, 1,
                    bptr, 1, l as isize,
                    0.0,
                    prod.as_mut_ptr(), tj as isize, 1,
                );
            }
            for ar in 0..ti {
                let gi = bi * t + ar;
                for bc in 0..tj {
                    let gj = bj * t + bc;
                    let v = prod[ar * tj + bc] as f64 * scale;
                    g[gi][gj] = v;
                    if gi != gj {
                        g[gj][gi] = v;
                    }
                }
            }
        }
    }
    Ok(g)
}

/// Exact f64 counterpart of [`gram_from_disk`] (the `--f64` build): identical
/// tiling, but the scratch holds f64 (8-byte stride) and the per-tile product is a
/// `dgemm` with f64 accumulation, so the gadget values are exact to ~1e-13 and the
/// distinct-off-diagonal-value count is trustworthy (no f32 merge/split). Doubles
/// the disk store and the resident tile bytes vs the f32 path, so callers should
/// halve `tile_rows` to keep the same peak RAM (two f64 tiles of 512 rows ≈ the
/// 16 GiB of two f32 tiles of 1024 rows at N=16).
pub fn gram_from_disk_f64(
    scratch: &ScratchW,
    ni: usize,
    l: usize,
    n: usize,
    tile_rows: usize,
) -> std::io::Result<Vec<Vec<f64>>> {
    let scale = 2.0 / (n * (n - 1) * dmin(n)) as f64;
    let mut g = vec![vec![0.0f64; ni]; ni];
    if ni == 0 || l == 0 {
        return Ok(g);
    }
    let t = tile_rows.max(1);
    let num_tiles = ni.div_ceil(t);
    let mut a = vec![0.0f64; t * l];
    let mut b = vec![0.0f64; t * l];
    let mut prod = vec![0.0f64; t * t];

    for bi in 0..num_tiles {
        let ti = t.min(ni - bi * t);
        scratch.read_rows_f64(bi * t, ti, l, &mut a[..ti * l])?;
        for bj in bi..num_tiles {
            let tj = t.min(ni - bj * t);
            let bptr = if bj == bi {
                a.as_ptr()
            } else {
                scratch.read_rows_f64(bj * t, tj, l, &mut b[..tj * l])?;
                b.as_ptr()
            };
            unsafe {
                matrixmultiply::dgemm(
                    ti, l, tj,
                    1.0,
                    a.as_ptr(), l as isize, 1,
                    bptr, 1, l as isize,
                    0.0,
                    prod.as_mut_ptr(), tj as isize, 1,
                );
            }
            for ar in 0..ti {
                let gi = bi * t + ar;
                for bc in 0..tj {
                    let gj = bj * t + bc;
                    let v = prod[ar * tj + bc] * scale;
                    g[gi][gj] = v;
                    if gi != gj {
                        g[gj][gi] = v;
                    }
                }
            }
        }
    }
    Ok(g)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decompose::{decompose_rep, dense_gadget, dense_gadget_matrix};
    use crate::lr_matrix::AdinkraRep;
    use crate::signed_perm::SignedPerm;

    fn cs_n4() -> AdinkraRep {
        let l = vec![
            SignedPerm::from_parts(vec![0, 1, 2, 3], vec![1, 1, 1, 1]).unwrap(),
            SignedPerm::from_parts(vec![1, 0, 3, 2], vec![1, -1, 1, -1]).unwrap(),
            SignedPerm::from_parts(vec![2, 3, 0, 1], vec![1, -1, -1, 1]).unwrap(),
            SignedPerm::from_parts(vec![3, 2, 1, 0], vec![1, 1, -1, -1]).unwrap(),
        ];
        AdinkraRep { n: 4, d: 4, l_matrices: l }
    }
    fn vs_n4() -> AdinkraRep {
        let l = vec![
            SignedPerm::from_parts(vec![0, 1, 2, 3], vec![1, 1, 1, 1]).unwrap(),
            SignedPerm::from_parts(vec![1, 0, 3, 2], vec![1, -1, -1, 1]).unwrap(),
            SignedPerm::from_parts(vec![2, 3, 0, 1], vec![1, 1, -1, -1]).unwrap(),
            SignedPerm::from_parts(vec![3, 2, 1, 0], vec![1, -1, 1, -1]).unwrap(),
        ];
        AdinkraRep { n: 4, d: 4, l_matrices: l }
    }

    /// The flat/Gram path must reproduce the trusted dense gadget exactly (within
    /// f32 round-off) on the N=4 irreducibles: self = 1, CS-VS = 0.
    #[test]
    fn flat_matches_dense_gadget_n4() {
        let cs = decompose_rep(&cs_n4()).unwrap();
        let vs = decompose_rep(&vs_n4()).unwrap();
        let s_cs = &cs.summands[0];
        let s_vs = &vs.summands[0];
        let dh_cs = DenseHoloraumy::from_summand(s_cs);
        let dh_vs = DenseHoloraumy::from_summand(s_vs);
        let w_cs = flatten_summand(s_cs);
        let w_vs = flatten_summand(s_vs);

        let pairs = [
            (gadget_from_flat(&w_cs, &w_cs, 4), dense_gadget(&dh_cs, &dh_cs)),
            (gadget_from_flat(&w_vs, &w_vs, 4), dense_gadget(&dh_vs, &dh_vs)),
            (gadget_from_flat(&w_cs, &w_vs, 4), dense_gadget(&dh_cs, &dh_vs)),
        ];
        for (flat, dense) in pairs {
            assert!(
                (flat - dense).abs() < 1e-6,
                "flat gadget {flat} != dense gadget {dense}"
            );
        }
        // self == 1, cross == 0 (the known N=4 values).
        assert!((gadget_from_flat(&w_cs, &w_cs, 4) - 1.0).abs() < 1e-6);
        assert!(gadget_from_flat(&w_cs, &w_vs, 4).abs() < 1e-6);
    }

    /// Block-diagonal direct sum of two N=4 reps (d=8), used to exercise the
    /// REAL decomposition path (summands with dense, non-(0,±1) restricted ops).
    fn block_diag_n4(a: &AdinkraRep, b: &AdinkraRep) -> AdinkraRep {
        let n = a.n;
        let (da, db) = (a.d, b.d);
        let mut l_matrices = Vec::with_capacity(n);
        for i in 0..n {
            let (la, lb) = (&a.l_matrices[i], &b.l_matrices[i]);
            let mut perm = vec![0u16; da + db];
            let mut sign = vec![0i8; da + db];
            for r in 0..da {
                perm[r] = la.perm[r];
                sign[r] = la.sign[r];
            }
            for r in 0..db {
                perm[da + r] = (da as u16) + lb.perm[r];
                sign[da + r] = lb.sign[r];
            }
            l_matrices.push(SignedPerm::from_parts(perm, sign).unwrap());
        }
        AdinkraRep { n, d: da + db, l_matrices }
    }

    /// Precondition for the Gram identity: every Vtilde_IJ of a decomposed summand
    /// must be antisymmetric (else Tr(A·B) != -<A,B>_F and the dot path is wrong).
    #[test]
    fn vtilde_is_antisymmetric_on_decomposed_summands() {
        let rep = block_diag_n4(&cs_n4(), &cs_n4()); // d=8 -> 2 dmin=4 summands
        let decomp = decompose_rep(&rep).unwrap();
        for s in &decomp.summands {
            let dh = DenseHoloraumy::from_summand(s);
            for (idx, m) in dh.vtilde.iter().enumerate() {
                let mt = m.transpose();
                // m + m^T == 0
                let worst = m.data.iter().zip(mt.data.iter())
                    .map(|(a, b)| (a + b).abs()).fold(0.0f64, f64::max);
                assert!(worst < 1e-9, "Vtilde[{idx}] not antisymmetric (worst {worst})");
            }
        }
    }

    /// gram_gadget_matrix must equal dense_gadget_matrix on a genuinely REDUCIBLE
    /// decomposition (CS⊕CS, d=8 -> 2 summands with dense restricted operators),
    /// not just on the trivial N=4 irreducibles.
    #[test]
    fn gram_matches_dense_on_reducible_decomposition() {
        let rep = block_diag_n4(&cs_n4(), &cs_n4());
        let decomp = decompose_rep(&rep).unwrap();
        assert_eq!(decomp.summands.len(), 2);
        let dense: Vec<DenseHoloraumy> =
            decomp.summands.iter().map(|s| DenseHoloraumy::from_summand(s)).collect();
        let dmat = dense_gadget_matrix(&dense);
        let vectors: Vec<Vec<f32>> =
            decomp.summands.iter().map(|s| flatten_summand(s)).collect();
        let gmat = gram_gadget_matrix(&vectors, 4);
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (gmat[i][j] - dmat[i][j]).abs() < 1e-5,
                    "reducible gram[{i}][{j}]={} != dense={}",
                    gmat[i][j], dmat[i][j]
                );
            }
            assert!((gmat[i][i] - 1.0).abs() < 1e-5, "summand self-gadget != 1");
        }
    }

    /// gram_gadget_matrix must equal dense_gadget_matrix on a small reducible
    /// stratum (CS + VS as separate irreducibles).
    #[test]
    fn gram_matrix_matches_dense_matrix() {
        let cs = decompose_rep(&cs_n4()).unwrap();
        let vs = decompose_rep(&vs_n4()).unwrap();
        let summands = [&cs.summands[0], &vs.summands[0]];
        let dense: Vec<DenseHoloraumy> =
            summands.iter().map(|s| DenseHoloraumy::from_summand(s)).collect();
        let dmat = dense_gadget_matrix(&dense);
        let vectors: Vec<Vec<f32>> = summands.iter().map(|s| flatten_summand(s)).collect();
        let gmat = gram_gadget_matrix(&vectors, 4);
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (gmat[i][j] - dmat[i][j]).abs() < 1e-6,
                    "gram[{i}][{j}]={} != dense={}",
                    gmat[i][j],
                    dmat[i][j]
                );
            }
        }
    }

    // -- disk-backed tiled Gram --------------------------------------------

    /// Write summands to a scratch file (summand g at offset g*l*4) and return it.
    fn write_scratch(summands: &[&IrrepSummand], l: usize, tag: &str) -> ScratchW {
        let path = std::env::temp_dir()
            .join(format!("adinkra_disktest_{}_{}.f32scratch", tag, std::process::id()));
        let sc = ScratchW::create(path, (summands.len() as u64) * (l as u64) * 4).unwrap();
        for (g, s) in summands.iter().enumerate() {
            sc.write_summand(g, &flatten_summand(s), l).unwrap();
        }
        sc.sync().unwrap();
        sc
    }

    /// Disk-tiled Gram must equal in-RAM gram and dense, with tile=1 (every block
    /// a single row/col), on the N=4 irreducibles.
    #[test]
    fn disk_tiled_matches_ram_and_dense_tile1() {
        let cs = decompose_rep(&cs_n4()).unwrap();
        let vs = decompose_rep(&vs_n4()).unwrap();
        let summands = [&cs.summands[0], &vs.summands[0]];
        let l = flat_len(4);
        let dense: Vec<DenseHoloraumy> =
            summands.iter().map(|s| DenseHoloraumy::from_summand(s)).collect();
        let dmat = dense_gadget_matrix(&dense);
        let vectors: Vec<Vec<f32>> = summands.iter().map(|s| flatten_summand(s)).collect();
        let gmat = gram_gadget_matrix(&vectors, 4);
        let sc = write_scratch(&summands, l, "t1");
        let dmat_disk = gram_from_disk(&sc, 2, l, 4, 1).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(dmat_disk[i][j], gmat[i][j], "disk != in-RAM gram at [{i}][{j}]");
                assert!((dmat_disk[i][j] - dmat[i][j]).abs() < 1e-6, "disk != dense at [{i}][{j}]");
            }
        }
        assert!((dmat_disk[0][0] - 1.0).abs() < 1e-6 && dmat_disk[0][1].abs() < 1e-6);
    }

    /// Partial-tile correctness: 3 summands (CS⊕CS -> 2 dense + VS) tiled by 2,
    /// so the last block is ragged in both axes. This is the offset/ragged-tile
    /// stress test. Also uses general-real (non 0,±1) restricted operators.
    #[test]
    fn disk_tiled_partial_tile_3summands_tile2() {
        let rep = block_diag_n4(&cs_n4(), &cs_n4());
        let decomp = decompose_rep(&rep).unwrap();
        let vs = decompose_rep(&vs_n4()).unwrap();
        let summands: Vec<&IrrepSummand> =
            vec![&decomp.summands[0], &decomp.summands[1], &vs.summands[0]];
        let l = flat_len(4);
        let vectors: Vec<Vec<f32>> = summands.iter().map(|s| flatten_summand(s)).collect();
        let gmat = gram_gadget_matrix(&vectors, 4);
        let sc = write_scratch(&summands, l, "t2");
        let dmat_disk = gram_from_disk(&sc, 3, l, 4, 2).unwrap();
        assert_eq!(dmat_disk.len(), 3);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(dmat_disk[i][j], gmat[i][j], "partial-tile disk != in-RAM at [{i}][{j}]");
                assert!((dmat_disk[i][j] - dmat_disk[j][i]).abs() < 1e-12, "asymmetry at [{i}][{j}]");
            }
            assert!((dmat_disk[i][i] - 1.0).abs() < 1e-5, "self-gadget[{i}] != 1");
        }
    }

    /// Offset round-trip: bytes read back at g*l*4 equal the written flat vector.
    #[test]
    fn disk_offset_round_trip() {
        let rep = block_diag_n4(&cs_n4(), &cs_n4());
        let decomp = decompose_rep(&rep).unwrap();
        let summands: Vec<&IrrepSummand> = decomp.summands.iter().collect();
        let l = flat_len(4);
        let flats: Vec<Vec<f32>> = summands.iter().map(|s| flatten_summand(s)).collect();
        let sc = write_scratch(&summands, l, "rt");
        for (g, flat) in flats.iter().enumerate() {
            let mut back = vec![0.0f32; l];
            sc.read_rows(g, 1, l, &mut back).unwrap();
            assert_eq!(&back, flat, "summand {g}: read-back != written flat vector");
        }
    }

    /// Scratch file is removed on drop (no leaked GiB-scale temp files).
    #[test]
    fn disk_scratch_removed_on_drop() {
        let cs = decompose_rep(&cs_n4()).unwrap();
        let summands = [&cs.summands[0]];
        let l = flat_len(4);
        let path = {
            let sc = write_scratch(&summands, l, "drop");
            let p = sc.path().to_path_buf();
            assert!(p.exists());
            p
        }; // sc dropped here
        assert!(!path.exists(), "scratch not removed on drop: {}", path.display());
    }

    // -- exact f64 disk-backed tiled Gram (--f64 build) --------------------

    /// Write summands as f64 (summand g at offset g*l*8) and return the scratch.
    fn write_scratch_f64(summands: &[&IrrepSummand], l: usize, tag: &str) -> ScratchW {
        let path = std::env::temp_dir()
            .join(format!("adinkra_disktest_{}_{}.f64scratch", tag, std::process::id()));
        let sc = ScratchW::create(path, (summands.len() as u64) * (l as u64) * 8).unwrap();
        let mut buf = vec![0.0f64; l];
        for (g, s) in summands.iter().enumerate() {
            flatten_summand_into_f64(s, &mut buf);
            sc.write_summand_f64(g, &buf, l).unwrap();
        }
        sc.sync().unwrap();
        sc
    }

    /// The f64 disk Gram must match the trusted dense f64 gadget to ~1e-12 (far
    /// tighter than the f32 path's ~1e-6), on a genuinely reducible decomposition
    /// (CS⊕CS -> 2 dense summands + VS) tiled by 2 to exercise ragged tiles.
    #[test]
    fn disk_f64_tiled_matches_dense() {
        let rep = block_diag_n4(&cs_n4(), &cs_n4());
        let decomp = decompose_rep(&rep).unwrap();
        let vs = decompose_rep(&vs_n4()).unwrap();
        let summands: Vec<&IrrepSummand> =
            vec![&decomp.summands[0], &decomp.summands[1], &vs.summands[0]];
        let l = flat_len(4);
        let dense: Vec<DenseHoloraumy> =
            summands.iter().map(|s| DenseHoloraumy::from_summand(s)).collect();
        let dmat = dense_gadget_matrix(&dense);
        let sc = write_scratch_f64(&summands, l, "f64t2");
        let disk = gram_from_disk_f64(&sc, 3, l, 4, 2).unwrap();
        assert_eq!(disk.len(), 3);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (disk[i][j] - dmat[i][j]).abs() < 1e-12,
                    "f64 disk[{i}][{j}]={} != dense={} (diff {:e})",
                    disk[i][j], dmat[i][j], (disk[i][j] - dmat[i][j]).abs()
                );
                assert!((disk[i][j] - disk[j][i]).abs() < 1e-13, "asymmetry at [{i}][{j}]");
            }
            assert!((disk[i][i] - 1.0).abs() < 1e-12, "f64 self-gadget[{i}] != 1");
        }
    }

    /// f64 offset round-trip: bytes read back at g*l*8 equal the written f64 flat
    /// vector exactly (no quantization).
    #[test]
    fn disk_f64_offset_round_trip() {
        let rep = block_diag_n4(&cs_n4(), &cs_n4());
        let decomp = decompose_rep(&rep).unwrap();
        let summands: Vec<&IrrepSummand> = decomp.summands.iter().collect();
        let l = flat_len(4);
        let sc = write_scratch_f64(&summands, l, "f64rt");
        let mut buf = vec![0.0f64; l];
        for (g, s) in summands.iter().enumerate() {
            let mut expect = vec![0.0f64; l];
            flatten_summand_into_f64(s, &mut expect);
            sc.read_rows_f64(g, 1, l, &mut buf).unwrap();
            assert_eq!(buf, expect, "summand {g}: f64 read-back != written flat vector");
        }
    }
}
