# Reproduction of the 10D supergravity L/R data

## Source

This reproduces the matrices described by Jacob Cigliano, Bergen Dahl, and
S. James Gates, Jr. in [*10D Supergravity Numerical Data Sets for L & R
Matrices*](https://arxiv.org/abs/2512.12157). The paper links the accompanying
[`mcmulaz/Super-Sym`](https://github.com/mcmulaz/Super-Sym) repository.

The reproduction pins:

- commit `8c8df92dac17853d7f6cb5b136ef2aec0efdea70`;
- source file `Garden Algebra`;
- source SHA-256 `d4499ad3077964b49659103fd3cc69e476ac8abe66a033f2284867a8e03b916c`;
- matrix token SHA-256 `4070fbeda9028cfd6c29097dfaf20df06300026e0c44d96e40c3d4b9a8faf244`.

The upstream repository supplies generative Wolfram Language source. It does
not supply literal matrix files. The primary generator and exact bosonic verifier
are implemented in Rust. The older NumPy and SymPy implementations remain only
as independent cross-checks. None of these implementations executes Wolfram Language.

## Basis and ordering

There are 16 colors. In the source convention:

- `L_I` has shape `82 x 176`, with bosonic rows and fermionic columns;
- `R_I` has shape `176 x 82`, with fermionic rows and bosonic columns;
- bosons are the 45 spatial `h_ij`, then the 36 spatial `B_ij`, then `phi`;
- fermions are `psi_mu(alpha)`, with `mu=0..9` outermost and `alpha=1..16`
  innermost, then `chi_1..chi_16`.

The sentence opening Section 6 reverses the L/R dimensions. Equation 6.0.8,
Equation 6.0.9, the conclusion, and the source establish the ordering above.

## Validation

Four validation components are retained:

1. `src/tendim_generate.rs` generates the complete artifact in Rust and verifies
   all 136 bosonic relations exactly over `Q(sqrt(2))`.
2. `scripts/gen_10d_data.py` independently evaluates the formulas numerically as
   a cross-check.
3. `scripts/eval_garden_exact.py` independently evaluates the same definitions
   with exact SymPy arithmetic and compares every matrix entry.
4. `src/tendim_data.rs` validates shapes, field order, allowed exact values,
   the complete matrix-content hash, every unordered Garden relation, and every
   displayed example in Eqs. 6.0.5-6.0.6.

Run:

```sh
cargo run --release -- tendim-generate
cargo run --release -- tendim-reproduce
cargo run --release -- tendim-convention-scan
python3 scripts/gen_10d_data.py /tmp/tendim-python.json
python3 scripts/eval_garden_exact.py
```

The complete audit gives:

| Check | Result |
|---|---:|
| L matrices | 16 at `82 x 176` |
| R matrices | 16 at `176 x 82` |
| Ordered L/R entries | 461,824 |
| Nonzero L/R entries | 3,040 / 7,648 |
| Matrix token hash | match |
| Unordered color pairs | 136 |
| Bosonic scalar equalities | 914,464 |
| Bosonic Garden residual | `1.6828760607268123e-12` maximum in `f64`; zero exactly |
| Fermionic pairs with nonzero `E_IJ` | 136 of 136 |
| Nonzero entries across unordered `E_IJ` | 132,352 at tolerance `1e-9` |
| Maximum `E_IJ` Frobenius norm | `9.924716620639604` |
| Maximum absolute `E_IJ` entry | `1` |

On an Apple M4 Pro with 48 GB memory, an optimized audit loaded the JSON in 12 ms
and validated its content in 4 ms. After three warmups, 30 full-algebra runs had
a 25.089 ms median and a 25.742 ms p95. These measurements are recorded in
`results/tendim_2512_reproduction_audit.json`; they are not performance guarantees.

## Printed examples

The paper/source comparison is conclusive about the existence of an
inconsistency, but not about the authors' intended final normalization.

- Four of the 13 displayed transformations match the source directly.
- Seven of the 26 displayed terms match directly.
- Eq. 6.0.5 mixes entries resembling colors 1 and 11, with two additional sign
  differences. Its `phi` line matches color 1.
- Eq. 6.0.6 has the source's field support. Its three graviton coefficients and
  three dilatino equations match, but six two-form coefficients do not.
- A fixed shared permutation of the displayed spinor labels cannot reconcile both
  samples. Eq. 6.0.6 fixes label 16 to 16, while Eq. 6.0.5 would require label 6
  to map to 16. This does not exclude a more general physical convention map.
- Equation 5.0.1 and the explicit sigma basis in Appendix A give
  `Q_1 h_11 = 2 psi_1(16)`, matching the generator rather than the displayed
  `2 psi_1(6)`.

The source also contains a normalization inconsistency: a comment states that
the reduced three-form contribution gives `1/8 MixedLeft`, while the executable
expression uses `1/16 MixedLeft`. The reproduction follows the executable source.
Author confirmation is required before changing that coefficient.

### Formula-level convention scan

Both explicit coefficient choices were generated as complete Rust datasets and
tested over `Q(sqrt(2))`:

| Branch | Exact failed pairs | Failed scalar entries | Paper equations matched |
|---|---:|---:|---:|
| Executable source, `1/16` | 0 of 136 | 0 | 4 of 13 |
| Source comment, `1/8` | 136 of 136 | 7,296 | 4 of 13 |

The `1/8` branch has maximum floating bosonic residual
`2.7386127875258306` and does not improve agreement with the printed examples.
With all other formulas fixed, it is not a Garden representation. This strongly
supports the executable `1/16` coefficient and indicates that the `1/8` comment
is stale or incomplete. It does not resolve the separate field-label and sign
discrepancies in Eq. 6.0.5 or the printed two-form coefficients in Eq. 6.0.6.

The machine-readable result is
`results/tendim_2512_convention_scan.json`.

## Result and boundary

The Rust generator and both independent Python cross-checks agree with the local
matrices entrywise, including the recorded basis, ordering, signs, and
normalizations. Rust verifies the complete bosonic Garden algebra exactly, and
the fermionic nonclosure data are generated for all unordered color pairs.

This does not establish that the executable three-form coefficient is the
authors' intended coefficient. It also does not construct an off-shell embedding.
The validated matrices are now a fixed input for an embedding search; candidate
extensions must reproduce this content hash and cancel the measured fermionic
nonclosure without disturbing the bosonic relations.

The ordered embedding and Adynkrafield equation program is maintained in
[`from-bbbm-closure-to-adynkrafield-equations.md`](from-bbbm-closure-to-adynkrafield-equations.md).
