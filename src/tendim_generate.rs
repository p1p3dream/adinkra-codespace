//! Rust generator for the pinned 10D supergravity L/R matrices.
//!
//! This is the primary implementation of the executable formulas in the
//! `Garden Algebra` source accompanying arXiv:2512.12157v1. The legacy NumPy
//! and SymPy ports remain as independent cross-checks.

#![allow(dead_code)]

use std::ops::{AddAssign, Mul};

use num_complex::Complex;
use num_rational::Ratio;
use num_traits::{ToPrimitive, Zero};
use serde::Serialize;

use crate::tendim_data::TenDimData;

type Rat = Ratio<i64>;
type Cx = Complex<Rat>;

fn rat(n: i64, d: i64) -> Rat {
    Ratio::new(n, d)
}

fn cx(n: i64) -> Cx {
    Complex::new(Ratio::from_integer(n), Rat::zero())
}

#[derive(Clone)]
struct CMat {
    rows: usize,
    cols: usize,
    data: Vec<Cx>,
}

impl CMat {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![Cx::zero(); rows * cols],
        }
    }

    fn from_i64(rows: usize, cols: usize, data: &[i64]) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self {
            rows,
            cols,
            data: data.iter().copied().map(cx).collect(),
        }
    }

    fn get(&self, row: usize, col: usize) -> &Cx {
        &self.data[row * self.cols + col]
    }

    fn add_scaled(&mut self, row: usize, col: usize, value: &Cx, scale: &Rat) {
        self.data[row * self.cols + col] += value.clone() * scale.clone();
    }

    fn scale(&self, scale: Rat) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .cloned()
                .map(|x| x * scale.clone())
                .collect(),
        }
    }

    fn sub(&self, other: &Self) -> Self {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols));
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| a.clone() - b.clone())
                .collect(),
        }
    }

    fn add_assign_scaled(&mut self, other: &Self, scale: Rat) {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols));
        for (out, value) in self.data.iter_mut().zip(&other.data) {
            *out += value.clone() * scale.clone();
        }
    }

    fn matmul(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows);
        let mut out = Self::zeros(self.rows, other.cols);
        for row in 0..self.rows {
            for inner in 0..self.cols {
                let a = self.get(row, inner);
                if a.is_zero() {
                    continue;
                }
                for col in 0..other.cols {
                    let b = other.get(inner, col);
                    if !b.is_zero() {
                        out.data[row * other.cols + col] += a.clone() * b.clone();
                    }
                }
            }
        }
        out
    }

    fn kron(&self, other: &Self) -> Self {
        let mut out = Self::zeros(self.rows * other.rows, self.cols * other.cols);
        for ar in 0..self.rows {
            for ac in 0..self.cols {
                let a = self.get(ar, ac);
                if a.is_zero() {
                    continue;
                }
                for br in 0..other.rows {
                    for bc in 0..other.cols {
                        out.data[(ar * other.rows + br) * out.cols + ac * other.cols + bc] =
                            a.clone() * other.get(br, bc).clone();
                    }
                }
            }
        }
        out
    }
}

fn kron4(a: &CMat, b: &CMat, c: &CMat, d: &CMat) -> CMat {
    a.kron(b).kron(c).kron(d)
}

fn pauli_tables() -> Vec<CMat> {
    let identity = CMat::from_i64(2, 2, &[1, 0, 0, 1]);
    let s1 = CMat::from_i64(2, 2, &[0, 1, 1, 0]);
    let mut s2 = CMat::zeros(2, 2);
    s2.data[1] = Complex::new(Rat::zero(), rat(-1, 1));
    s2.data[2] = Complex::new(Rat::zero(), rat(1, 1));
    let s3 = CMat::from_i64(2, 2, &[1, 0, 0, -1]);
    vec![
        kron4(&identity, &identity, &identity, &identity),
        kron4(&s2, &s2, &s2, &s2),
        kron4(&s2, &s2, &identity, &s1),
        kron4(&s2, &s2, &identity, &s3),
        kron4(&s2, &s1, &s2, &identity),
        kron4(&s2, &s3, &s2, &identity),
        kron4(&s2, &identity, &s1, &s2),
        kron4(&s2, &identity, &s3, &s2),
        kron4(&s1, &identity, &identity, &identity),
        kron4(&s3, &identity, &identity, &identity),
    ]
}

fn eta(mu: usize) -> i64 {
    if mu == 0 { -1 } else { 1 }
}

fn sig2_up(sig: &[CMat], mu: usize, nu: usize) -> CMat {
    let left = sig[mu].matmul(&sig[nu].scale(Ratio::from_integer(eta(nu))));
    let right = sig[nu].matmul(&sig[mu].scale(Ratio::from_integer(eta(mu))));
    left.sub(&right).scale(rat(1, 2))
}

fn tilde_sig2_up(sig: &[CMat], mu: usize, nu: usize) -> CMat {
    let left = sig[mu].scale(Ratio::from_integer(eta(mu))).matmul(&sig[nu]);
    let right = sig[nu].scale(Ratio::from_integer(eta(nu))).matmul(&sig[mu]);
    left.sub(&right).scale(rat(1, 2))
}

fn sig3_up(sig: &[CMat], mu: usize, nu: usize, rho: usize) -> CMat {
    const PERMS: [([usize; 3], i64); 6] = [
        ([0, 1, 2], 1),
        ([0, 2, 1], -1),
        ([1, 0, 2], -1),
        ([1, 2, 0], 1),
        ([2, 0, 1], 1),
        ([2, 1, 0], -1),
    ];
    let values = [mu, nu, rho];
    let mut out = CMat::zeros(16, 16);
    for (perm, sign) in PERMS {
        let a = values[perm[0]];
        let b = values[perm[1]];
        let c = values[perm[2]];
        let term = sig[a]
            .matmul(&sig[b].scale(Ratio::from_integer(eta(b))))
            .matmul(&sig[c]);
        out.add_assign_scaled(&term, rat(sign, 6));
    }
    out
}

struct Tables {
    sig_up: Vec<CMat>,
    sigma: Vec<CMat>,
    sig2: Vec<Vec<CMat>>,
    tilde2: Vec<Vec<CMat>>,
    sig3_0: Vec<Vec<CMat>>,
    mixed: Vec<Vec<Vec<CMat>>>,
}

fn tables() -> Tables {
    let sig_up = pauli_tables();
    let sigma = (0..10)
        .map(|mu| sig_up[mu].scale(Ratio::from_integer(eta(mu))))
        .collect::<Vec<_>>();
    let sig2 = (0..10)
        .map(|mu| {
            (0..10)
                .map(|nu| sig2_up(&sig_up, mu, nu).scale(Ratio::from_integer(eta(mu) * eta(nu))))
                .collect()
        })
        .collect();
    let tilde2 = (0..10)
        .map(|mu| (0..10).map(|nu| tilde_sig2_up(&sig_up, mu, nu)).collect())
        .collect();
    let sig3_0 = (0..10)
        .map(|rho| {
            (0..10)
                .map(|xi| sig3_up(&sig_up, 0, rho, xi))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mixed = (0..10)
        .map(|mu| {
            (0..10)
                .map(|rho| {
                    (0..10)
                        // SigmaTilde[mu] equals sigUp[mu] in the pinned source.
                        .map(|xi| sig_up[mu].matmul(&sig3_0[rho][xi]))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    Tables {
        sig_up,
        sigma,
        sig2,
        tilde2,
        sig3_0,
        mixed,
    }
}

fn h_order() -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for mu in 1..=9 {
        for nu in mu..=9 {
            out.push((mu, nu));
        }
    }
    out
}

fn b_order() -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for mu in 1..=9 {
        for nu in (mu + 1)..=9 {
            out.push((mu, nu));
        }
    }
    out
}

fn add_h(row: &mut [Cx], order: &[(usize, usize)], key: (usize, usize), value: Cx) {
    if let Some(index) = order.iter().position(|&x| x == key) {
        row[index] += value;
    }
}

fn add_b(row: &mut [Cx], order: &[(usize, usize)], key: (usize, usize), value: Cx) {
    if let Some(index) = order.iter().position(|&x| x == key) {
        row[45 + index] += value;
    }
}

fn real(value: &Cx) -> f64 {
    assert!(
        value.im.is_zero(),
        "generated matrix has an imaginary entry"
    );
    let raw = value.re.to_f64().unwrap();
    let rounded = (raw * 1e12).round() / 1e12;
    if rounded == 0.0 { 0.0 } else { rounded }
}

/// Generate all sixteen L and R matrices from the pinned formulas.
pub fn generate() -> TenDimData {
    let t = tables();
    let hs = h_order();
    let bs = b_order();
    let mut ls = Vec::with_capacity(16);
    let mut rs = Vec::with_capacity(16);

    for charge in 0..16 {
        let mut l = vec![vec![0.0; 176]; 82];
        for (row_index, &(mu, nu)) in hs.iter().enumerate() {
            for spinor in 0..16 {
                l[row_index][nu * 16 + spinor] += real(t.sigma[mu].get(charge, spinor));
                l[row_index][mu * 16 + spinor] += real(t.sigma[nu].get(charge, spinor));
            }
        }
        for (offset, &(mu, nu)) in bs.iter().enumerate() {
            let row = 45 + offset;
            for spinor in 0..16 {
                l[row][nu * 16 + spinor] += real(t.sigma[mu].get(charge, spinor));
                l[row][mu * 16 + spinor] -= real(t.sigma[nu].get(charge, spinor));
                l[row][160 + spinor] += real(t.sig2[mu][nu].get(charge, spinor));
            }
        }
        l[81][160 + charge] = round12(8.0_f64.sqrt());

        let mut r = vec![vec![0.0; 82]; 176];
        for mu in 0..10 {
            for dotted in 0..16 {
                let mut row = vec![Cx::zero(); 82];
                for rho in 0..10 {
                    let hkey = if rho <= mu { (rho, mu) } else { (mu, rho) };
                    add_h(
                        &mut row,
                        &hs,
                        hkey,
                        t.tilde2[0][rho].get(dotted, charge).clone() * rat(-1, 2),
                    );
                    if rho != mu {
                        let (bkey, sign) = if rho < mu {
                            ((rho, mu), 1)
                        } else {
                            ((mu, rho), -1)
                        };
                        add_b(
                            &mut row,
                            &bs,
                            bkey,
                            t.tilde2[0][rho].get(dotted, charge).clone() * rat(-sign, 2),
                        );
                    }
                }
                for rho in 0..10 {
                    for xi in (rho + 1)..10 {
                        add_b(
                            &mut row,
                            &bs,
                            (rho, xi),
                            t.mixed[mu][rho][xi].get(dotted, charge).clone() * rat(1, 16),
                        );
                    }
                }
                r[mu * 16 + dotted] = row.iter().map(real).collect();
            }
        }
        for spinor in 0..16 {
            let mut row = vec![Cx::zero(); 82];
            for rho in 0..10 {
                for xi in (rho + 1)..10 {
                    add_b(
                        &mut row,
                        &bs,
                        (rho, xi),
                        t.sig3_0[rho][xi].get(spinor, charge).clone() * rat(-1, 8),
                    );
                }
            }
            let mut out = row.iter().map(real).collect::<Vec<_>>();
            out[81] = real(t.sig_up[0].get(spinor, charge)) / 8.0_f64.sqrt();
            out[81] = round12(out[81]);
            r[160 + spinor] = out;
        }
        ls.push(l);
        rs.push(r);
    }

    TenDimData {
        nb: 82,
        nf: 176,
        n: 16,
        source: "arXiv:2512.12157 / github.com/mcmulaz/Super-Sym 'Garden Algebra' Mathematica code, evaluated numerically".to_string(),
        l: ls,
        r: rs,
    }
}

fn round12(value: f64) -> f64 {
    let value = (value * 1e12).round() / 1e12;
    if value == 0.0 { 0.0 } else { value }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Quad {
    rational: Rat,
    sqrt2: Rat,
}

impl Quad {
    fn zero() -> Self {
        Self {
            rational: Rat::zero(),
            sqrt2: Rat::zero(),
        }
    }

    fn rational(n: i64, d: i64) -> Self {
        Self {
            rational: rat(n, d),
            sqrt2: Rat::zero(),
        }
    }

    fn sqrt2(n: i64, d: i64) -> Self {
        Self {
            rational: Rat::zero(),
            sqrt2: rat(n, d),
        }
    }
}

impl AddAssign for Quad {
    fn add_assign(&mut self, rhs: Self) {
        self.rational += rhs.rational;
        self.sqrt2 += rhs.sqrt2;
    }
}

impl Mul for Quad {
    type Output = Quad;

    fn mul(self, rhs: Self) -> Self::Output {
        Quad {
            rational: self.rational.clone() * rhs.rational.clone()
                + rat(2, 1) * self.sqrt2.clone() * rhs.sqrt2.clone(),
            sqrt2: self.rational * rhs.sqrt2 + self.sqrt2 * rhs.rational,
        }
    }
}

fn exact_l(value: f64) -> Quad {
    for (candidate, exact) in [
        (-2.0, Quad::rational(-2, 1)),
        (-1.0, Quad::rational(-1, 1)),
        (0.0, Quad::zero()),
        (1.0, Quad::rational(1, 1)),
        (2.0, Quad::rational(2, 1)),
        (8.0_f64.sqrt(), Quad::sqrt2(2, 1)),
    ] {
        if (value - candidate).abs() <= 1e-9 {
            return exact;
        }
    }
    panic!("illegal generated L value {value}");
}

fn exact_r(value: f64) -> Quad {
    for (candidate, exact) in [
        (-0.5, Quad::rational(-1, 2)),
        (-0.4375, Quad::rational(-7, 16)),
        (-0.125, Quad::rational(-1, 8)),
        (-0.0625, Quad::rational(-1, 16)),
        (0.0, Quad::zero()),
        (0.0625, Quad::rational(1, 16)),
        (0.125, Quad::rational(1, 8)),
        (0.4375, Quad::rational(7, 16)),
        (0.5, Quad::rational(1, 2)),
        (1.0 / 8.0_f64.sqrt(), Quad::sqrt2(1, 4)),
    ] {
        if (value - candidate).abs() <= 1e-9 {
            return exact;
        }
    }
    panic!("illegal generated R value {value}");
}

fn sparse_rows(matrix: &[Vec<f64>], side: char) -> Vec<Vec<(usize, Quad)>> {
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .filter_map(|(column, &value)| {
                    let exact = if side == 'L' {
                        exact_l(value)
                    } else {
                        exact_r(value)
                    };
                    (exact != Quad::zero()).then_some((column, exact))
                })
                .collect()
        })
        .collect()
}

fn sparse_product(
    left: &[Vec<(usize, Quad)>],
    right: &[Vec<(usize, Quad)>],
    cols: usize,
) -> Vec<Quad> {
    let mut out = vec![Quad::zero(); left.len() * cols];
    for (row, entries) in left.iter().enumerate() {
        for (inner, a) in entries {
            for (col, b) in &right[*inner] {
                out[row * cols + *col] += a.clone() * b.clone();
            }
        }
    }
    out
}

/// Verify all 136 bosonic Garden relations over Q(sqrt(2)).
pub fn verify_exact_bosonic(data: &TenDimData) -> usize {
    let ls = data
        .l
        .iter()
        .map(|matrix| sparse_rows(matrix, 'L'))
        .collect::<Vec<_>>();
    let rs = data
        .r
        .iter()
        .map(|matrix| sparse_rows(matrix, 'R'))
        .collect::<Vec<_>>();
    let mut checked = 0;
    for i in 0..16 {
        for j in i..16 {
            let a = sparse_product(&ls[i], &rs[j], 82);
            let b = sparse_product(&ls[j], &rs[i], 82);
            for row in 0..82 {
                for col in 0..82 {
                    let mut value = a[row * 82 + col].clone();
                    value += b[row * 82 + col].clone();
                    let expected = if i == j && row == col {
                        Quad::rational(2, 1)
                    } else {
                        Quad::zero()
                    };
                    assert_eq!(
                        value, expected,
                        "exact Garden failure at ({i},{j},{row},{col})"
                    );
                }
            }
            checked += 1;
        }
    }
    checked
}

pub fn max_entrywise_difference(left: &TenDimData, right: &TenDimData) -> f64 {
    let mut max = 0.0_f64;
    for i in 0..16 {
        for (a, b) in left.l[i].iter().flatten().zip(right.l[i].iter().flatten()) {
            max = max.max((a - b).abs());
        }
        for (a, b) in left.r[i].iter().flatten().zip(right.r[i].iter().flatten()) {
            max = max.max((a - b).abs());
        }
    }
    max
}

fn normalize_json_spacing(input: &str) -> String {
    let mut out = String::with_capacity(input.len() + input.len() / 4);
    let mut in_string = false;
    let mut escaped = false;
    for ch in input.chars() {
        if in_string {
            out.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
        } else {
            match ch {
                '"' => {
                    in_string = true;
                    out.push(ch);
                }
                ',' | ':' => {
                    out.push(ch);
                    out.push(' ');
                }
                c if c.is_whitespace() => {}
                _ => out.push(ch),
            }
        }
    }
    out
}

#[derive(Serialize)]
struct Relations<'a> {
    bosonic: &'a str,
    fermionic: &'a str,
}

/// Serialize the generated matrices using the committed artifact's stable JSON
/// spacing and the checked-in provenance manifest.
pub fn artifact_json(data: &TenDimData) -> String {
    let manifest = include_str!("../data/tendim_10d_provenance.json");
    let relations = Relations {
        bosonic: "L_I R_J + L_J R_I = 2 delta_IJ I_82",
        fermionic: "R_I L_J + R_J L_I = 2 delta_IJ I_176 + 2 E_IJ",
    };
    let source = serde_json::to_string(&data.source).unwrap();
    let relations = normalize_json_spacing(&serde_json::to_string(&relations).unwrap());
    let provenance = normalize_json_spacing(manifest);
    let l = normalize_json_spacing(&serde_json::to_string(&data.l).unwrap());
    let r = normalize_json_spacing(&serde_json::to_string(&data.r).unwrap());
    format!(
        "{{\"nb\": 82, \"nf\": 176, \"n\": 16, \"source\": {source}, \"relations\": {relations}, \"provenance\": {provenance}, \"L\": {l}, \"R\": {r}}}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dataset_path() -> String {
        format!("{}/data/tendim_10d_lr.json", env!("CARGO_MANIFEST_DIR"))
    }

    #[test]
    fn rust_generator_matches_every_committed_entry() {
        let generated = generate();
        let committed = crate::tendim_data::load(&dataset_path());
        assert_eq!(max_entrywise_difference(&generated, &committed), 0.0);
    }

    #[test]
    fn rust_generator_closes_exactly_over_q_sqrt2() {
        let generated = generate();
        assert_eq!(verify_exact_bosonic(&generated), 136);
    }

    #[test]
    fn rust_serialization_matches_committed_artifact_bytes() {
        let generated = generate();
        let expected = std::fs::read_to_string(dataset_path()).unwrap();
        assert_eq!(artifact_json(&generated), expected);
    }
}
