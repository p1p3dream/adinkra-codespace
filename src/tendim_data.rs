//! Loader and verifier for the published 10D supergravity L/R matrix dataset.
//!
//! Source: arXiv:2512.12157 "10D Supergravity Numerical Data Sets for L & R
//! Matrices" and its data repository (GitHub `mcmulaz/Super-Sym`, the "Garden
//! Algebra" Mathematica file). The original repository stores *generative*
//! Mathematica code (sigma-matrix Kronecker products), not literal matrices;
//! `data/tendim_10d_lr.json` is the numerically evaluated result of a Python
//! re-implementation of that code (`scripts/gen_10d_data.py`).
//!
//! PROVENANCE CAVEAT (from adversarial review): because the dataset is
//! REGENERATED (not a literal download), satisfying the bosonic Garden relation
//! is necessary-but-not-sufficient evidence of fidelity (a different basis /
//! color ordering / sign convention could satisfy it too). Safe claim: "this
//! JSON satisfies the bosonic Garden relation to ~1.7e-12 with a nonzero
//! fermionic remnant." NOT yet safe: "byte-faithful to the published matrices."
//! Full fidelity needs a pinned upstream commit, a Wolfram re-export, and exact
//! entry/hash comparison. The generator script is checked in for reproducibility.
//!
//! Shapes: 16 generators. L_I is nb x nf (82 x 176), R_I is nf x nb (176 x 82).
//! nb = 82 bosons, nf = 176 fermions. This is a non-valise, non-square
//! representation.
//!
//! The matrices satisfy:
//!   - Bosonic (closes):     L_I R_J + L_J R_I = 2 delta_IJ I_82
//!   - Fermionic (on-shell): R_I L_J + R_J L_I = 2 delta_IJ I_176 + 2 E_IJ
//! where E_IJ is the on-shell remnant / non-closure tensor (nonzero, including
//! on the diagonal, because the algebra closes only up to equations of motion).
//!
//! This module is intentionally self-contained: matrix multiply is hand-rolled
//! (no external linear-algebra crate). It is NOT registered in main.rs.

use serde::Deserialize;

/// Raw parsed dataset. `L[i]` is a 82x176 matrix (Vec of 82 rows, each 176
/// long). `R[i]` is a 176x82 matrix. Entries are stored as f64 (the real parts;
/// the published matrices are real even though intermediate sigma matrices are
/// complex).
#[derive(Debug, Clone, Deserialize)]
pub struct TenDimData {
    pub nb: usize,
    pub nf: usize,
    pub n: usize,
    #[serde(default)]
    pub source: String,
    #[serde(rename = "L")]
    pub l: Vec<Vec<Vec<f64>>>,
    #[serde(rename = "R")]
    pub r: Vec<Vec<Vec<f64>>>,
}

/// Load the dataset from a JSON file produced from the Super-Sym Garden Algebra
/// code. Panics with a descriptive message on IO / parse / shape errors.
pub fn load(path: &str) -> TenDimData {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read 10D dataset {path}: {e}"));
    let data: TenDimData = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("failed to parse 10D dataset {path}: {e}"));
    data.validate_shapes();
    data
}

impl TenDimData {
    /// Sanity-check declared dimensions against the actual parsed matrices.
    pub fn validate_shapes(&self) {
        assert_eq!(self.l.len(), self.n, "expected {} L matrices", self.n);
        assert_eq!(self.r.len(), self.n, "expected {} R matrices", self.n);
        for (i, li) in self.l.iter().enumerate() {
            assert_eq!(li.len(), self.nb, "L[{i}] should have {} rows", self.nb);
            for (row_idx, row) in li.iter().enumerate() {
                assert_eq!(
                    row.len(),
                    self.nf,
                    "L[{i}] row {row_idx} should have {} cols",
                    self.nf
                );
            }
        }
        for (i, ri) in self.r.iter().enumerate() {
            assert_eq!(ri.len(), self.nf, "R[{i}] should have {} rows", self.nf);
            for (row_idx, row) in ri.iter().enumerate() {
                assert_eq!(
                    row.len(),
                    self.nb,
                    "R[{i}] row {row_idx} should have {} cols",
                    self.nb
                );
            }
        }
    }

    /// Verify both Garden relations directly from the parsed matrices.
    ///
    /// Returns `(bosonic_residual, fermionic_e_norm)` where:
    /// - `bosonic_residual` is the maximum, over all ordered pairs (I, J), of
    ///   the Frobenius norm of `L_I R_J + L_J R_I - 2 delta_IJ I_82`. For a
    ///   valid dataset this is ~0.
    /// - `fermionic_e_norm` is the maximum, over all pairs (I, J), of the
    ///   Frobenius norm of the remnant `E_IJ = (R_I L_J + R_J L_I)/2 - delta_IJ
    ///   I_176`. This is generically nonzero (the 10D on-shell remnant).
    pub fn verify_garden(&self) -> (f64, f64) {
        let mut bosonic_residual = 0.0_f64;
        let mut fermionic_e_norm = 0.0_f64;

        for i in 0..self.n {
            for j in 0..self.n {
                // Bosonic: L_I R_J + L_J R_I vs 2 delta_IJ I_82  (82x82)
                let lirj = matmul(&self.l[i], &self.r[j]); // 82x82
                let ljri = matmul(&self.l[j], &self.r[i]); // 82x82
                let delta = if i == j { 2.0 } else { 0.0 };
                let res = frob_diff_minus_scaled_identity(&lirj, &ljri, delta, self.nb);
                if res > bosonic_residual {
                    bosonic_residual = res;
                }

                // Fermionic remnant: E_IJ = (R_I L_J + R_J L_I)/2 - delta_IJ I
                let rilj = matmul(&self.r[i], &self.l[j]); // 176x176
                let rjli = matmul(&self.r[j], &self.l[i]); // 176x176
                let e_norm = e_remnant_norm(&rilj, &rjli, i == j, self.nf);
                if e_norm > fermionic_e_norm {
                    fermionic_e_norm = e_norm;
                }
            }
        }

        (bosonic_residual, fermionic_e_norm)
    }

    /// Frobenius norm of the remnant E_IJ for a single ordered pair (I, J).
    pub fn e_remnant(&self, i: usize, j: usize) -> f64 {
        let rilj = matmul(&self.r[i], &self.l[j]);
        let rjli = matmul(&self.r[j], &self.l[i]);
        e_remnant_norm(&rilj, &rjli, i == j, self.nf)
    }
}

/// Dense matrix multiply: (m x k) * (k x n) -> (m x n). Hand-rolled, no deps.
fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let k = a[0].len();
    let n = b[0].len();
    debug_assert_eq!(b.len(), k, "inner dimensions must agree");
    let mut out = vec![vec![0.0_f64; n]; m];
    for i in 0..m {
        let a_row = &a[i];
        let out_row = &mut out[i];
        for p in 0..k {
            let aip = a_row[p];
            if aip == 0.0 {
                continue;
            }
            let b_row = &b[p];
            for j in 0..n {
                out_row[j] += aip * b_row[j];
            }
        }
    }
    out
}

/// Frobenius norm of `(x + y) - scale*I` for square `dim x dim` matrices.
fn frob_diff_minus_scaled_identity(
    x: &[Vec<f64>],
    y: &[Vec<f64>],
    scale: f64,
    dim: usize,
) -> f64 {
    let mut acc = 0.0_f64;
    for i in 0..dim {
        for j in 0..dim {
            let mut v = x[i][j] + y[i][j];
            if i == j {
                v -= scale;
            }
            acc += v * v;
        }
    }
    acc.sqrt()
}

/// Frobenius norm of E_IJ = (rilj + rjli)/2 - delta_IJ I.
fn e_remnant_norm(rilj: &[Vec<f64>], rjli: &[Vec<f64>], is_diag: bool, dim: usize) -> f64 {
    let mut acc = 0.0_f64;
    for i in 0..dim {
        for j in 0..dim {
            let mut e = 0.5 * (rilj[i][j] + rjli[i][j]);
            if is_diag && i == j {
                e -= 1.0;
            }
            acc += e * e;
        }
    }
    acc.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dataset_path() -> String {
        // Resolve relative to the crate root regardless of test cwd.
        format!("{}/data/tendim_10d_lr.json", env!("CARGO_MANIFEST_DIR"))
    }

    #[test]
    fn tendim_loads_with_expected_shapes() {
        let d = load(&dataset_path());
        assert_eq!(d.nb, 82);
        assert_eq!(d.nf, 176);
        assert_eq!(d.n, 16);
    }

    #[test]
    fn tendim_bosonic_garden_relation_holds() {
        let d = load(&dataset_path());
        let (bosonic_residual, fermionic_e_norm) = d.verify_garden();
        // Bosonic relation must close (allow tiny float noise).
        assert!(
            bosonic_residual < 1e-9,
            "bosonic Garden residual too large: {bosonic_residual}"
        );
        // Fermionic remnant E_IJ is the on-shell term: it must be nonzero,
        // confirming this is the genuine non-closing 10D representation.
        assert!(
            fermionic_e_norm > 1e-6,
            "expected nonzero fermionic E remnant, got {fermionic_e_norm}"
        );
    }
}
