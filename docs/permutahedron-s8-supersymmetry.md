# Six signed eight-color systems

## Result

The six signed `8 x 8` matrix systems `CC`, `CT`, `CV`, `TT`, `TV`, and `VV`
from Ref. [1] have been reconstructed in Rust from the printed permutations and
Boolean factors. The calculation evaluates all 16 residue classes
`(m mod 4, n mod 4)` for each system containing the height parameters in
Eq. (2.5). Including the three parameter-free systems, this gives 51 signed
matrix branches.

The result reproduces the split reported in Ref. [1]:

| System | Printed status | Garden closure | HYMN trace | Valid valise Adinkra |
|---|---|---:|---:|---:|
| `CC` | nonclosure | no | 16 | no |
| `CT` | closure | yes | 0 | yes |
| `CV` | closure | yes | 0 | yes |
| `TT` | nonclosure | no for all 16 branches | 16 | no |
| `TV` | nonclosure | no for all 16 branches | 16 | no |
| `VV` | nonclosure | no for all 16 branches | 16 | no |

The HYMN matrix agrees with Eq. (3.3) on all 51 branches. It is
`sigma_3 tensor I_8` for `CT` and `CV`, and `I_16` for `CC`, `TT`, `TV`, and
`VV`.

## Matrix and closure calculation

For each color, the signed matrix is

```text
L_I = S_I P_I,
R_I = L_I^T,
```

where `P_I` is the printed permutation matrix and `S_I` is the diagonal matrix
defined by the printed eight-bit Boolean factor. Bit zero controls matrix row
one.

Both Garden relations are checked entry by entry:

```text
L_I R_J + L_J R_I = 2 delta_IJ I_8,
R_I L_J + R_J L_I = 2 delta_IJ I_8.
```

The calculation checks 417,792 dense matrix entries. There are zero residual
entries for `CT` and `CV`. The 49 nonclosing branches contain 12,416 residual
entries in total. Each residual is an integer, so the calculation uses no
floating-point tolerance.

The lower three systems have two residual patterns across their 16 parameter
branches:

- 4 nonclosing color pairs, 128 residual entries, and minimum Garden distance
  16;
- 12 nonclosing color pairs, 384 residual entries, and minimum Garden distance
  18.

## Graph construction

Every branch produces a signed eight-color bipartite graph with eight bosons,
eight fermions, 64 colored edges, and 112 two-color squares. The program checks
the dashing parity of every square.

The `CT` and `CV` graphs satisfy Garden closure and all 112 odd-dashing
conditions. They are therefore stored as valise Adinkras.

The other 49 graphs retain the published signed matrices and nonclosure. They
are stored as signed colored valise graphs and are not relabeled as off-shell
Adinkras.

## Sector recognition

The calculation compares several finite invariants and diagnostics.

### HYMN

The HYMN class is the published separator and matches the closure split on
every branch:

```text
trace(C_hat) = 0   for CT and CV,
trace(C_hat) = 16  for CC, TT, TV, and VV.
```

### Garden distance

Every unsigned `R8` octet admits 524,288 Garden signings. The printed sign
choice lies in that affine solution space for `CT` and `CV`. For the other four
systems, the nearest Garden signing differs from the printed signs by 16 or 18
edge signs, depending on the height-parameter branch.

This distance measures the defect of the printed sign choice relative to
Garden closure. Distance zero is equivalent to closure here. It is not an
independent physical invariant.

### Unsigned coset data

Left-right coset coincidence does not separate the two closing systems from the
four nonclosing systems. `TV` is not left-right coincident, but its printed
signs do not close.

The unsigned Bruhat characteristic polynomial and the canonical
chromocharacter value separate the closing and nonclosing sets in these six
examples. This is an observed property of the six published octets, not a
general classification theorem.

### Gadget boundary

The holoraumy gadget is a representation invariant for the two closing
systems. The same matrix expression is retained for the four nonclosing graphs
only as a formal diagnostic. It is not reported as a supersymmetry
representation invariant for those graphs.

## Reproduction

From the repository root:

```sh
cargo run --release -- perm-s8-susy-build
cargo run --release -- perm-s8-susy-verify
node scripts/test_permutahedron_s8_supersymmetry.mjs
```

The Rust generator writes:

- `data/permutahedron_s8_supersymmetry.json`
- `results/permutahedron_s8_supersymmetry_validation.json`

Validated artifact hashes:

| Artifact | SHA256 |
|---|---|
| `data/permutahedron_s8_supersymmetry.json` | `3df539d4d82472566cc4ab90dc5dbf5f8ab1035ee9708cd1d7c26a51a550d241` |
| `results/permutahedron_s8_supersymmetry_validation.json` | `a6fb4d1ec4e6571d55d9780db81906f589aa1fc354bc9c6b19d00048f41002b4` |

The independent JavaScript audit reconstructs every signed matrix from the
stored source permutations and Boolean factors. It then recomputes both Garden
relations, every HYMN product, and the graph counts for all 51 branches.

Serve the repository root with a local static-file server and open
`visualizer/permutahedron_s8_supersymmetry.html` to inspect the six systems,
their parameter branches, signed matrices, graphs, closure residuals, HYMN
classes, and finite permutation data.

## Boundaries

- This is a reproduction and finite extension of the calculations in Ref. [1].
  It is not a new off-shell construction.
- The 16 residue classes exhaust the signs generated by the integer parameters
  in Eq. (2.5). They do not introduce continuous deformations.
- A different Garden signing on the same unsigned octet does not change the
  closure status of the printed four-dimensional construction.
- The result supplies a checked finite atlas for these six systems. It does not
  derive the sought field equation.

## Primary reference

1. D. D. Bristow, J. H. Caporaletti, A. J. Cianciara, S. J. Gates Jr.,
   D. Levine, and G. Yerger, "A Note On Exemplary Off-Shell Constructions Of
   4D, N=2 Supersymmetry Representations,"
   [arXiv:2012.14015v7](https://arxiv.org/abs/2012.14015).
