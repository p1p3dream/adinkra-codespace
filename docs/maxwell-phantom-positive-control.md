# Maxwell phantom-sector positive control

## Result

The magnetic phantom sector of the four-dimensional Maxwell field-strength
multiplet has been extracted from the verified chiral-vector transformations.
The calculation uses the four first-supersymmetry charges in arXiv:1405.0048
and the phantom definitions in Section 5 of arXiv:0907.3605.

The source field basis is

```text
(E1, E2, E3, d | B1, B2, B3)
```

with four gaugino components. The electric fields and auxiliary field remain on
the worldline. The three magnetic fields vanish under zero-brane reduction and
form the phantom sector.

## Exact checks

The implementation constructs:

- the `7 x 4` transpose of the fermion-up linkage for each of four charges;
- the temporal boson-down linkage;
- the phantom matrix `P_A = u_tilde_A^T - Delta_A^0` from Eq. (5.1); and
- all three spatial boson-down linkages.

The printed reduction phase `lambda = i Psi` is applied before comparing the up
and down linkages. The result is:

| Check | Result |
|---|---:|
| visible worldline bosons | 4 |
| magnetic phantoms | 3 |
| nonzero phantom entries | 12 |
| nonphantom rows of `P_A` | all zero |
| temporal down-links from magnetic fields | all zero |
| residual entries in Eq. (5.8) | 0 |

Equation (5.8),

```text
(Delta_A^a)_(B^b) = epsilon_(b a c) (Delta_A^0)_(E_c),
```

is checked on every spatial direction, charge, magnetic row, and fermion
component using exact Gaussian-integer arithmetic.

## Reproduction

```sh
cargo run --release -- maxwell-phantom-build
cargo run --release -- maxwell-phantom-verify
cargo test --release maxwell_phantom
node scripts/test_maxwell_phantom.mjs
```

The JavaScript cross-check independently constructs the gamma matrices and all
four linkage tensors. It compares every entry with the Rust artifact.

Artifacts and source hashes:

| Item | SHA256 |
|---|---|
| `results/maxwell_phantom.json` | `d991c03b3e1336b1411e62a8319f96f978ae79edd3bbbc38a0ebaef30afcc984` |
| arXiv:0907.3605 PDF | `720e737ac980b346d41daac97219ec23f29fcdc044b471c3a292eaa808997668` |
| arXiv:1405.0048 PDF | `8e666e70c9484033e1223fc80b16a5db562c0ec4e499721962277f6a3987ae20` |

## Consequence

The previously omitted gauge and phantom sector is now explicit for the
Maxwell positive control. The next gate is the canonical Bianchi reshuffling
of Eqs. (5.4)-(5.5), followed by the complete gauge-enhancement condition in
Eq. (5.11). Passing that gate on the known Maxwell shadow is required before
testing eight-color candidates.

## Boundary

This result verifies the phantom support and Eq. (5.8) in one fixed source
basis. It does not yet verify the complete hatted Omega condition in Eq. (5.11)
and does not establish gauge enhancement for any `VM1`, `VM2`, or `VM3`
construction.
