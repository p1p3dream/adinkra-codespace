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

use crate::decompose::{DenseHoloraumy, IrrepSummand};
use crate::holoraumy::dmin;
use rayon::prelude::*;

/// Length of a flattened holoraumy vector: C(n,2) * dmin(n)^2.
pub fn flat_len(n: usize) -> usize {
    (n * (n - 1) / 2) * dmin(n) * dmin(n)
}

/// f32 bytes of the full flat-vector store for `num_irreps` summands at `n`.
pub fn flat_store_bytes(n: usize, num_irreps: usize) -> u64 {
    (num_irreps as u64) * (flat_len(n) as u64) * 4
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

/// Symmetric gadget matrix from in-RAM flat f32 vectors, computed in parallel
/// over rows (upper triangle then mirrored), f64 accumulation. Extra memory is
/// just the result matrix; the flat vectors are borrowed.
pub fn gram_gadget_matrix(vectors: &[Vec<f32>], n: usize) -> Vec<Vec<f64>> {
    let m = vectors.len();
    // Upper triangle per row, in parallel.
    let mut mat: Vec<Vec<f64>> = (0..m)
        .into_par_iter()
        .map(|i| {
            let mut row = vec![0.0f64; m];
            for j in i..m {
                row[j] = gadget_from_flat(&vectors[i], &vectors[j], n);
            }
            row
        })
        .collect();
    // Mirror the lower triangle from the (already-computed) upper triangle.
    for i in 0..m {
        for j in 0..i {
            mat[i][j] = mat[j][i];
        }
    }
    mat
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
}
