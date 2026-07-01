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
//! (no external linear-algebra crate) and SHA-256 is vendored below (no new
//! dependency added). It is registered in main.rs.

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

    /// STRICT allowed-value-set validator.
    ///
    /// Maps every L entry to one of the exact L tokens and every R entry to one
    /// of the exact R tokens, within `VALUE_TOL` (1e-9). The allowed token sets
    /// are the Codex-confirmed value facts for this dataset:
    ///   - L in { -2, -1, 0, 1, 2, sqrt8 (= 2*sqrt2) }
    ///   - R in { 0, +-1/16, +-1/8, +-7/16, +-1/2, +-1/sqrt8 }
    ///
    /// Returns `Ok(())` if every entry matches a token, otherwise `Err` listing
    /// the first offending entries (matrix index + position + raw value).
    pub fn validate_value_sets(&self) -> Result<(), String> {
        let mut errors: Vec<String> = Vec::new();
        const MAX_REPORTED: usize = 32;

        for (mi, m) in self.l.iter().enumerate() {
            for (ri, row) in m.iter().enumerate() {
                for (ci, &v) in row.iter().enumerate() {
                    if l_token(v).is_none() {
                        if errors.len() < MAX_REPORTED {
                            errors.push(format!("L[{mi}][{ri}][{ci}] = {v:?} not an allowed L token"));
                        }
                    }
                }
            }
        }
        for (mi, m) in self.r.iter().enumerate() {
            for (ri, row) in m.iter().enumerate() {
                for (ci, &v) in row.iter().enumerate() {
                    if r_token(v).is_none() {
                        if errors.len() < MAX_REPORTED {
                            errors.push(format!("R[{mi}][{ri}][{ci}] = {v:?} not an allowed R token"));
                        }
                    }
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "value-set validation failed ({} offending entr{}, showing up to {}):\n{}",
                errors.len(),
                if errors.len() == 1 { "y" } else { "ies" },
                MAX_REPORTED,
                errors.join("\n")
            ))
        }
    }

    /// CANONICAL SYMBOLIC TOKEN HASH.
    ///
    /// Maps each matrix entry to its exact symbolic token string (e.g. "-2",
    /// "0", "sqrt8", "-7/16", "1/sqrt8"), then serializes L followed by R as a
    /// single canonical token stream in fixed order (matrix index, row, col;
    /// tokens separated by ',', matrices/rows by structural markers) and hashes
    /// it with a vendored SHA-256.
    ///
    /// This hash depends ONLY on matrix CONTENT (the symbolic value of every
    /// entry, in order). It is invariant to:
    ///   - JSON metadata changes (source/relations/nb/nf/n field text),
    ///   - float formatting (0.5 vs 5e-1 vs 0.50000000001 within tol),
    /// because entries are first snapped to exact tokens.
    ///
    /// Panics if any entry is not an allowed token (call `validate_value_sets`
    /// first to get a descriptive error instead).
    pub fn canonical_token_hash(&self) -> String {
        let mut stream = String::with_capacity(self.n * self.nb * self.nf * 2);
        // Fixed structural framing so different shapes can never collide.
        stream.push_str("TENDIM-TOKENS-v1;");
        stream.push('L');
        for m in &self.l {
            stream.push('|');
            for row in m {
                for &v in row {
                    let t = l_token(v).unwrap_or_else(|| {
                        panic!("canonical_token_hash: L entry {v:?} is not an allowed token")
                    });
                    stream.push_str(t);
                    stream.push(',');
                }
                stream.push(';');
            }
        }
        stream.push_str("#R");
        for m in &self.r {
            stream.push('|');
            for row in m {
                for &v in row {
                    let t = r_token(v).unwrap_or_else(|| {
                        panic!("canonical_token_hash: R entry {v:?} is not an allowed token")
                    });
                    stream.push_str(t);
                    stream.push(',');
                }
                stream.push(';');
            }
        }
        sha256_hex(stream.as_bytes())
    }
}

/// Tolerance for snapping a float entry to an exact symbolic token.
const VALUE_TOL: f64 = 1e-9;

/// sqrt(8) = 2*sqrt(2) (appears in L).
const SQRT8: f64 = 2.8284271247461903; // (2.0 * std::f64::consts::SQRT_2)
/// 1/sqrt(8) (appears in R).
const INV_SQRT8: f64 = 0.35355339059327373; // 1.0 / SQRT8

/// Map an L entry to its canonical token string, or None if it matches none.
fn l_token(v: f64) -> Option<&'static str> {
    // (value, token) table for L, EXACTLY the documented set { -2,-1,0,1,2,sqrt8 }.
    // The dataset uses only +sqrt8 (the Phi row); -sqrt8 is intentionally NOT
    // admitted so the validator matches the documented value set strictly.
    const TABLE: &[(f64, &str)] = &[
        (-2.0, "-2"),
        (-1.0, "-1"),
        (0.0, "0"),
        (1.0, "1"),
        (2.0, "2"),
        (SQRT8, "sqrt8"),
    ];
    snap(v, TABLE)
}

/// Map an R entry to its canonical token string, or None if it matches none.
fn r_token(v: f64) -> Option<&'static str> {
    // (value, token) table for R: { 0, +-1/16, +-1/8, +-7/16, +-1/2, +-1/sqrt8 }.
    const TABLE: &[(f64, &str)] = &[
        (0.0, "0"),
        (0.0625, "1/16"),
        (-0.0625, "-1/16"),
        (0.125, "1/8"),
        (-0.125, "-1/8"),
        (0.4375, "7/16"),
        (-0.4375, "-7/16"),
        (0.5, "1/2"),
        (-0.5, "-1/2"),
        (INV_SQRT8, "1/sqrt8"),
        (-INV_SQRT8, "-1/sqrt8"),
    ];
    snap(v, TABLE)
}

/// Find the token whose value is within `VALUE_TOL` of `v`.
fn snap(v: f64, table: &[(f64, &'static str)]) -> Option<&'static str> {
    for &(val, tok) in table {
        if (v - val).abs() <= VALUE_TOL {
            return Some(tok);
        }
    }
    None
}

/// Tolerances for [`verify_lift`] (Default: all 1e-9).
#[derive(Debug, Clone)]
pub struct LiftTolerances {
    pub bosonic: f64,
    pub entrywise: f64,
    pub partner: f64,
}
impl Default for LiftTolerances {
    fn default() -> Self {
        LiftTolerances { bosonic: 1e-9, entrywise: 1e-9, partner: 1e-9 }
    }
}

/// Outcome of [`verify_lift`].
#[derive(Debug, Clone)]
pub struct LiftVerification {
    pub shape_ok: bool,
    /// Candidate's own bosonic Garden residual (must be < tol.bosonic).
    pub bosonic_residual: f64,
    /// ENTRYWISE max |candidate - target| over every L and R entry (must be
    /// < tol.entrywise). Anchored entrywise, NOT per-pair Frobenius norm, which is
    /// orthogonally degenerate and admits false positives (adversarial review).
    pub max_lr_diff: f64,
    pub value_set_ok: bool,
    pub partner_ok: bool,
    /// Worst |L_I R_I − I_nb| entry over all I.
    pub partner_worst: f64,
    /// `Some(true/false)` when a `pinned_hash` was supplied; `None` otherwise.
    pub token_hash_match: Option<bool>,
    pub passed: bool,
}

/// Verify a candidate 10D lift against a reference `target` dataset: entrywise L/R
/// match, candidate bosonic Garden closure, value-set legality, the partner
/// relation `L_I R_I = I_nb`, and (optionally) the exact `canonical_token_hash`.
///
/// HONEST SCOPE (do not overclaim): with no lift PRODUCER yet, this is a REGRESSION
/// ORACLE. Exercised on the dataset's own L/R it certifies self-consistency of the
/// known 10D N=1 SUPERGRAVITY multiplet (82×176; NOT super-Yang-Mills, NOT
/// basis-independent). A pass means "entrywise-matches the pinned reference", never
/// "a new higher-dimensional multiplet was constructed or discovered". The token
/// hash pins only this dataset's specific basis/ordering.
pub fn verify_lift(
    candidate: &TenDimData,
    target: &TenDimData,
    tol: &LiftTolerances,
    pinned_hash: Option<&str>,
) -> LiftVerification {
    // Shape: counts AND per-matrix dims (nb×nf for L, nf×nb for R), so the
    // entrywise comparison below is index-safe and never panics.
    let dims_ok = |mats: &[Vec<Vec<f64>>], rows: usize, cols: usize| {
        mats.iter().all(|m| m.len() == rows && m.iter().all(|row| row.len() == cols))
    };
    // Validate BOTH candidate and target dims (a caller may pass a malformed
    // target), so the entrywise zip below is always index-safe.
    let shape_ok = candidate.nb == target.nb
        && candidate.nf == target.nf
        && candidate.n == target.n
        && candidate.l.len() == target.l.len()
        && candidate.r.len() == target.r.len()
        && candidate.l.len() == candidate.n
        && candidate.r.len() == candidate.n
        && dims_ok(&candidate.l, candidate.nb, candidate.nf)
        && dims_ok(&candidate.r, candidate.nf, candidate.nb)
        && dims_ok(&target.l, target.nb, target.nf)
        && dims_ok(&target.r, target.nf, target.nb);
    if !shape_ok {
        return LiftVerification {
            shape_ok: false,
            bosonic_residual: f64::INFINITY,
            max_lr_diff: f64::INFINITY,
            value_set_ok: false,
            partner_ok: false,
            partner_worst: f64::INFINITY,
            token_hash_match: pinned_hash.map(|_| false),
            passed: false,
        };
    }

    let (bosonic_residual, _fermionic) = candidate.verify_garden();

    let mut max_lr_diff = 0.0f64;
    for i in 0..candidate.l.len() {
        for (a, b) in candidate.l[i].iter().flatten().zip(target.l[i].iter().flatten()) {
            max_lr_diff = max_lr_diff.max((a - b).abs());
        }
        for (a, b) in candidate.r[i].iter().flatten().zip(target.r[i].iter().flatten()) {
            max_lr_diff = max_lr_diff.max((a - b).abs());
        }
    }

    let value_set_ok = candidate.validate_value_sets().is_ok();

    // Partner relation L_I R_I = I_nb.
    let mut partner_worst = 0.0f64;
    for i in 0..candidate.n {
        let prod = matmul(&candidate.l[i], &candidate.r[i]); // nb×nb
        for (a, row) in prod.iter().enumerate() {
            for (b, &v) in row.iter().enumerate() {
                let want = if a == b { 1.0 } else { 0.0 };
                partner_worst = partner_worst.max((v - want).abs());
            }
        }
    }
    let partner_ok = partner_worst < tol.partner;

    // The token hash is exact-content; only meaningful (and only non-panicking)
    // when every entry snaps to a legal token, i.e. value_set_ok.
    let token_hash_match = match pinned_hash {
        Some(h) if value_set_ok => Some(candidate.canonical_token_hash() == h),
        Some(_) => Some(false), // illegal values -> cannot match the pinned hash
        None => None,
    };

    let passed = shape_ok
        && bosonic_residual < tol.bosonic
        && max_lr_diff < tol.entrywise
        && value_set_ok
        && partner_ok
        && token_hash_match != Some(false);

    LiftVerification {
        shape_ok,
        bosonic_residual,
        max_lr_diff,
        value_set_ok,
        partner_ok,
        partner_worst,
        token_hash_match,
        passed,
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

// ---------------------------------------------------------------------------
// Vendored SHA-256 (FIPS 180-4). Self-contained, no external dependency, so the
// canonical token hash needs no new crate. Operates on a byte slice and returns
// the lowercase hex digest. Not constant-time (not needed: this hashes public
// matrix content for a content-lock, not secrets).
// ---------------------------------------------------------------------------
fn sha256_hex(input: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    // Pre-processing: padding.
    let bit_len = (input.len() as u64).wrapping_mul(8);
    let mut msg = input.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    // Process each 512-bit chunk.
    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut hh = h[7];

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = String::with_capacity(64);
    for word in h {
        out.push_str(&format!("{word:08x}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// verify_lift as a regression oracle: the dataset self-verifies (entrywise +
    /// hash), and any perturbation is rejected.
    #[test]
    fn verify_lift_self_and_perturbation() {
        const PINNED: &str =
            "4070fbeda9028cfd6c29097dfaf20df06300026e0c44d96e40c3d4b9a8faf244";
        let data = load(&dataset_path());
        let v = verify_lift(&data, &data, &LiftTolerances::default(), Some(PINNED));
        assert!(v.passed, "dataset must self-verify: {v:?}");
        assert_eq!(v.token_hash_match, Some(true));
        assert!(v.max_lr_diff == 0.0 && v.partner_ok && v.value_set_ok);

        // Perturb one L entry: breaks entrywise, value-set, bosonic, and hash.
        let mut bad = load(&dataset_path());
        bad.l[0][0][0] += 0.1;
        let vb = verify_lift(&bad, &data, &LiftTolerances::default(), Some(PINNED));
        assert!(!vb.passed, "perturbed candidate must be rejected: {vb:?}");
        assert!(vb.max_lr_diff > 1e-3, "entrywise diff should catch the perturbation");
    }

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

    /// Vendored SHA-256 sanity check against published FIPS-180-4 test vectors,
    /// so a hash regression is caught independently of the dataset.
    #[test]
    fn vendored_sha256_matches_known_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b"The quick brown fox jumps over the lazy dog"),
            "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592"
        );
    }

    /// Full-quantity loader/verifier test: shapes, dof counts, value-set
    /// validation, bosonic closure, fermionic remnant (computed-here), a
    /// rank/partner sanity check, and the canonical-token hash lock.
    #[test]
    fn tendim_full_quantity_validation() {
        let d = load(&dataset_path());

        // --- Shapes: 16 generators; L is 82x176, R is 176x82. ---
        assert_eq!(d.n, 16, "expected 16 generators");
        assert_eq!(d.l.len(), 16);
        assert_eq!(d.r.len(), 16);
        for li in &d.l {
            assert_eq!(li.len(), 82, "each L matrix has 82 rows");
            assert!(li.iter().all(|row| row.len() == 176), "each L row has 176 cols");
        }
        for ri in &d.r {
            assert_eq!(ri.len(), 176, "each R matrix has 176 rows");
            assert!(ri.iter().all(|row| row.len() == 82), "each R row has 82 cols");
        }

        // --- DOF counts: nb=82=55+45+1-19, nf=176=160+16. ---
        assert_eq!(d.nb, 82, "boson dof");
        assert_eq!(d.nf, 176, "fermion dof");
        assert_eq!(55 + 45 + 1 - 19, 82, "boson dof decomposition");
        assert_eq!(160 + 16, 176, "fermion dof decomposition");

        // --- Strict allowed-value-set validation. ---
        d.validate_value_sets()
            .expect("all L/R entries must map to allowed symbolic tokens");

        // --- Bosonic residual must close; fermionic remnant computed here. ---
        let (bosonic_residual, fermionic_e_norm) = d.verify_garden();
        assert!(
            bosonic_residual < 1e-9,
            "bosonic Garden residual too large: {bosonic_residual}"
        );
        // NOTE: fermionic_e_norm is OUR COMPUTED on-shell remnant from this JSON,
        // NOT a paper-published number. We only assert it is nonzero (the algebra
        // closes only up to equations of motion) and report the value.
        assert!(
            fermionic_e_norm > 1e-6,
            "expected nonzero COMPUTED-HERE fermionic E remnant, got {fermionic_e_norm}"
        );
        eprintln!(
            "[tendim] COMPUTED-HERE max ||E_IJ|| (Frobenius) = {fermionic_e_norm} \
             (our remnant, not paper-reported); bosonic residual = {bosonic_residual}"
        );

        // --- Rank / partner sanity: R is the genuine GR-partner of L, not an
        // arbitrary matrix that happens to satisfy the bosonic side. No L_I and
        // no R_I may be the zero matrix (a trivial rank-deficient generator),
        // and every boson row of L_I must reach at least one fermion (no boson
        // is left without any partner coupling for generator I). ---
        for (gi, li) in d.l.iter().enumerate() {
            assert!(
                li.iter().any(|row| row.iter().any(|&x| x.abs() > VALUE_TOL)),
                "L[{gi}] is entirely zero (degenerate generator)"
            );
            for (rrow, row) in li.iter().enumerate() {
                assert!(
                    row.iter().any(|&x| x.abs() > VALUE_TOL),
                    "L[{gi}] boson row {rrow} has no fermion partner"
                );
            }
        }
        for (gi, ri) in d.r.iter().enumerate() {
            assert!(
                ri.iter().any(|row| row.iter().any(|&x| x.abs() > VALUE_TOL)),
                "R[{gi}] is entirely zero (degenerate generator)"
            );
        }
        // Diagonal-block partner relation (the strong partner test): for I==I
        // the bosonic product L_I R_I must equal I_82 (2 delta / 2), i.e. R_I is
        // a right-partner of L_I on the boson space.
        for gi in 0..d.n {
            let prod = matmul(&d.l[gi], &d.r[gi]); // 82x82, must be I_82
            let mut max_off = 0.0_f64;
            let mut min_diag = f64::INFINITY;
            for r in 0..d.nb {
                for c in 0..d.nb {
                    if r == c {
                        min_diag = min_diag.min(prod[r][c]);
                    } else {
                        max_off = max_off.max(prod[r][c].abs());
                    }
                }
            }
            assert!(
                max_off < 1e-9 && (min_diag - 1.0).abs() < 1e-9,
                "L[{gi}] R[{gi}] is not I_82 (max_off={max_off}, min_diag={min_diag})"
            );
        }

        // --- Hash lock: canonical token hash must equal the pinned constant. ---
        // Content-only: invariant to JSON metadata + float formatting.
        const PINNED_HASH: &str =
            "4070fbeda9028cfd6c29097dfaf20df06300026e0c44d96e40c3d4b9a8faf244";
        let h = d.canonical_token_hash();
        eprintln!("[tendim] canonical_token_hash = {h}");
        assert_eq!(
            h, PINNED_HASH,
            "canonical token hash drift: dataset CONTENT changed (pinned={PINNED_HASH}, got={h})"
        );
    }
}
