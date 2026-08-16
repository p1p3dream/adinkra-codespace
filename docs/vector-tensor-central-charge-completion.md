# Vector-tensor central-charge completion

## Result

The published `TV` S8 system is not a failed multiplet. Its ordinary Garden
residual contains an exact one-generator central extension on 8 of its 16 sign
branches.

The accepted branches are exactly

```text
m + n = 1 mod 2
```

For every accepted branch, only four unordered color pairs are nonzero:

| Color pair | Coefficient of `2 Z` |
|---|---:|
| `(1,7)` | `+1` |
| `(2,8)` | `+1` |
| `(3,5)` | `-1` |
| `(4,6)` | `-1` |

The paired bosonic and fermionic residual tensor has exact rank one. The
extracted operators `Z_B` and `Z_F` are symmetric signed permutations, square
to the identity, have trace zero, and intertwine all eight supercharges:

```text
L_I Z_F = Z_B L_I
R_I Z_B = Z_F R_I
```

The full central-charge-aware closure identity passes for every `I,J`:

```text
L_I R_J + L_J R_I = 2 delta_IJ I + 2 Omega_IJ Z_B
R_I L_J + R_J L_I = 2 delta_IJ I + 2 Omega_IJ Z_F
```

The 8 even-parity `TV` branches have paired residual rank 6 and are not a
one-central-charge realization.

## Source bridge

The component vector-tensor system and its worldline basis are printed in
[arXiv:1405.0048](https://arxiv.org/abs/1405.0048), Sec. 3.6, Eqs. (76)-(84),
and Appendix F. The physical N=2 algebra and nontrivial central-charge action
are given in
[arXiv:hep-th/9609016](https://arxiv.org/abs/hep-th/9609016), especially Eqs.
(2.1), (3.34), (4.3), and (4.6).

The implementation independently reconstructs all Appendix F matrices from
the printed 4x4 Boolean factors, permutation cycles, and `a+`, `a-`, `b+`,
`b-` signs. It checks 8,192 entries over all 16 `(m mod 4,n mod 4)` branches.
There are zero mismatches with the committed `TV` fixture.

The parity result also follows directly from the source coefficients:

```text
c1 = cos((m+n) pi/2)
s1 = sin((m-n) pi/2)
c2+ = cos(m pi) + cos(n pi)
c2- = cos(m pi) - cos(n pi)
```

When `m+n` is odd, `c1=c2+=0`. The surviving bosonic and fermionic residuals
each use one node operator, producing the rank-one completion. When `m+n` is
even, several independent operators survive.

## Physical worldline interpretation

At zero spatial momentum and in temporal gauge, Eq. (4.6) has the structure

```text
Z phi = D
Z D = d_t^2 phi
Z V_i = d_t H_i
Z H_i = d_t V_i
```

Here `H_i` is a normalized spatial dual of the two-form. Lower the auxiliary
with `D = d_t F`. In the valised bosonic basis

```text
(phi, V1, V2, V3 | F, H1, H2, H3)
```

the central charge is

```text
Z = d_t (sigma_1 tensor I_4).
```

Its node operator is a trace-zero involution with eigenvalue multiplicities
`4+4`. Every accepted extracted `Z_B` has exactly this signed-permutation
conjugacy signature. The artifact also stores an explicit signed-permutation
conjugator taking every extracted `Z_B` to `sigma_1 tensor I_4`, rather than
relying only on the signature. Its color coefficient matrix is checked directly
against the surviving Eq. (81) Pauli tensor, up to the overall sign absorbed
into the definition of `Z`.

The companion 4D fixture now fixes the source normalization rather than choosing
it dynamically:

```text
phi_1405 = phi_960
A_1405 = V_960
B_1405 = B_960 / 6
d_1405 = -D_960
```

With `D_960 = d_t F`, the semantic bosonic basis above conjugates the extracted
`TV:m1n0` operator exactly to `sigma_1 tensor I_4`. The Weyl-to-Majorana phase
map also reproduces all 64 entries of `Z_F`. The component central action and
the color tensor are simultaneously oriented against the committed convention,
so their product in the extended algebra agrees exactly. Mutating the `1/6`
two-form normalization is rejected.

Gauge-orbit preservation is explicit. `Z A` depends only on `H=dB`, and `Z B`
depends only on `F=dA`. At strict zero spatial momentum, temporal gauge is
preserved. Residual time-independent gauge transformations act trivially on the
spatial nodes, while lowering the auxiliary requires a declared treatment of
the time-independent integration constant.

## Full six-sector census

The exact residual scan covers all 51 published branches:

| Sector | Ordinary Garden | One central charge | Higher-rank residual |
|---|---:|---:|---:|
| `CC` | 0 | 1 | 0 |
| `CT` | 1 | 0 | 0 |
| `CV` | 1 | 0 | 0 |
| `TT` | 0 | 8 | 8 |
| `TV` | 0 | 8 | 8 |
| `VV` | 0 | 8 | 8 |
| **Total** | **2** | **25** | **24** |

For `TT` and `VV`, the one-charge branches have `m+n` even. For `TV`, they
have `m+n` odd. Ordinary zero-central-charge closure therefore discarded real
algebraic structure in half of each nonclosing parameter family.

## Enriched equivalence

The 25 one-charge branches were also classified as enriched tuples

```text
(L, R, Z_B, Z_F, Omega).
```

The declared finite equivalence allows independent signed boson and fermion
permutations, signed color permutations, and the simultaneous orientation
change `(Z,Omega) -> (-Z,-Omega)`. Boson-fermion duality is not needed.

All 25 branches form one exact enriched equivalence class. The artifact stores
25 direct witnesses and verifies 1,600 L entries plus 600 `Z_B`, `Z_F`, and
`Omega` entries. Flipping only the central orientation in a witness is rejected.
Thus `CC`, central `TT`, central `TV`, and central `VV` do not define distinct
worldline one-Z classes under this policy.

## Full R8 atlas projection

The common enriched one-Z class was transported through all 30 conjugate R8
families and every right coset.

| Check | Result |
|---|---:|
| subgroup families | 30 |
| relabeling paths | 1,209,600 |
| distinct supports | 151,200 |
| paths per support | exactly 8 |
| transported entries checked | 96,768,000 |
| direct family representative algebra checks | 30 |

Every one of the 151,200 unsigned hyperedges admits an exact one-central-charge
signing. The existing ordinary Garden transport independently covers the same
151,200 supports. Therefore every unsigned support admits both algebra types:

1. ordinary Garden closure with central rank zero;
2. one-generator central closure with central rank one.

This closes the hypergraph question sharply. Unsigned R8 support cannot
determine even the worldline algebra type, so it cannot identify a unique 4D
parent. The central operator is real higher-dimensional information, but it must
be retained as signed operator data rather than projected down to an unsigned
hyperedge.

## Reproduction

```bash
cargo test vector_tensor_central_charge --no-fail-fast
cargo test vector_tensor_4d --no-fail-fast
cargo run -- vector-tensor-central-charge-build
cargo run -- vector-tensor-central-charge-verify
cargo run -- vector-tensor-4d-build
cargo run -- vector-tensor-4d-verify
cargo run -- vector-tensor-central-equivalence-build
cargo run -- vector-tensor-central-equivalence-verify
cargo run -- vector-tensor-central-atlas-build
cargo run -- vector-tensor-central-atlas-verify
```

Artifacts:

- `data/vector_tensor_central_charge.json`
- `results/vector_tensor_central_charge_validation.json`
- `data/vector_tensor_4d.json`
- `results/vector_tensor_4d_validation.json`
- `results/vector_tensor_central_equivalence.json`
- `results/vector_tensor_central_atlas.json`

## Current boundary

The companion 4D fixture proves all 720 corrected Eq. (78) component relations,
including explicit vector and two-form gauge residues. The fixed Eq. (4.6)
bosonic, fermionic, and simultaneous `Z`/`Omega` zero-brane bridge is exact.

The remaining source-level gate is narrower: directly reduce the repaired
hep-th/9609016 Eq. (4.5) transformations to the 512 Appendix F `L/R` entries.
That transcription must certify the three required SU(2)-index repairs and the
coherent spatial frame, epsilon, and supercharge conventions. Until it passes,
the implementation claims an exact central bridge and exact closure on the
1405 fixture side, not complete term-by-term equivalence of the two published
4D presentations.
