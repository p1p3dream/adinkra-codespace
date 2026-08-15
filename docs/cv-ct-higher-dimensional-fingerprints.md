# Four-dimensional fingerprints retained above the worldline reduction

## Result

The verified chiral-vector and chiral-tensor transformation systems have been
decomposed into their algebraic, temporal-derivative, spatial-derivative, and
gauge-residue parts. The comparison is exact and uses the source field bases
and conventions of arXiv:1405.0048.

Both systems reduce to `8|8` Garden representations. Their higher-dimensional
data are not the same.

| Quantity | Chiral-vector | Chiral-tensor |
|---|---:|---:|
| raw component slots | 17 | 19 |
| transformation relations | 136 | 152 |
| transformation terms | 328 | 344 |
| algebraic terms | 72 | 88 |
| temporal-derivative terms | 64 | 64 |
| spatial-derivative terms | 192 | 192 |
| relations containing spatial derivatives | 88 | 80 |
| gauge-potential form degree | 1 | 2 |
| field-strength form degree | 2 | 3 |
| gauge-parameter form degree | 0 | 1 |
| potential components before temporal gauge | 4 | 6 |
| components removed by temporal gauge | 1 | 3 |
| potential components after temporal gauge | 3 | 3 |
| nonzero gauge-residue relations | 96 | 176 |
| gauge-residue terms | 128 | 384 |

The equal count of 192 spatial-derivative terms does not make the spatial
operators equal. Their exact coefficients, target fields, source fields, and
derivative directions differ. Canonical source-basis serializations give
different SHA256 values:

| Operator | Chiral-vector | Chiral-tensor |
|---|---|---|
| spatial linkage | `0d477476173478eaa1597aa71651177142c2809a9993dcc4f3934cbbe48c76e0` | `2670ba5f78f83126be375c8799ab9ff0f812a6d016f658b659f8974b1b9bc383` |
| gauge residue | `5ca5a1a3bdb62f0752d2e1b3d59ca9475bc68aa0f8855a92edc7565df765a7aa` | `6a3086cbf284517a5fe9ab70c9070dbfd7f384f77bec48628d90dc17f43b0504` |

These hashes are reproducibility identifiers for the stated source bases.
They are not basis-independent physical invariants.

## What survives and what is lost

The temporal reductions retain eight bosonic and eight fermionic nodes in
both cases. They discard the distinction between:

- a vector potential and a two-form potential;
- a scalar gauge parameter and a one-form gauge parameter;
- a two-form field strength and a three-form field strength;
- one removed temporal component and three removed temporal components; and
- the two complete spatial linkage operators.

The reduced signed matrices are therefore insufficient input for assigning a
four-dimensional parent.

## Minimum data required for another candidate

A meaningful higher-dimensional test of `VM1`, `VM2`, or `VM3` requires:

1. a Lorentz representation for every bosonic and fermionic component;
2. complete spatial-derivative linkage coefficients in a stated basis;
3. the gauge-potential form degree and complete gauge transformation;
4. the gauge-invariant field strength and any Bianchi identity;
5. the temporal gauge, surviving zero modes, and field-to-node map; and
6. closure of every supercharge pair on every component, including gauge
   residues.

The one-dimensional matrices do not determine these data. They must be given
by a higher-dimensional ansatz or source construction before a closure test is
defined.

## Reproduction

```sh
cargo run --release -- higher-dimensional-fingerprint-build
cargo run --release -- higher-dimensional-fingerprint-verify
cargo test --release higher_dimensional_fingerprint
```

Artifact:

- `results/higher_dimensional_fingerprint.json`
- SHA256: `4a17adf58ac988eeef54892fff964375bf76149619ca1de09d95f00256771805`

## Boundary

This comparison records the information lost in the two verified reductions.
It does not assign a four-dimensional parent to `VM1`, `VM2`, or `VM3`. It
also does not establish that the listed data are sufficient for a unique
higher-dimensional reconstruction.
