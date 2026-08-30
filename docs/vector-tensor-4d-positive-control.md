# Vector-tensor 4D positive control

## Result

The `(m,n)=(1,0)` vector-tensor transformations from
[arXiv:1405.0048](https://arxiv.org/abs/1405.0048), Eq. (77), reproduce the
complete corrected Eq. (78) closure ledger exactly.

The fixture retains all potential and gauge terms. It checks:

| Component class | Relations |
|---|---:|
| scalar `phi` | 36 |
| vector potential `A_mu` | 144 |
| two-form potential `B_mu_nu` | 216 |
| auxiliary `d` | 36 |
| fermion doublet `lambda_a^i` | 288 |
| **Total** | **720** |

All 720 exact sparse-polynomial relations pass with zero residual terms.

## Physical branch

For `m=1,n=0`, the source coefficients are

```text
a = i sigma_2
b = identity_2
c1 = 0
s1 = 1
c2+ = 0
c2- = -2
```

This is one of the eight odd-parity branches independently shown to have exact
one-central-charge worldline closure.

## Source corrections and convention translation

Two source defects are recorded rather than silently ignored:

1. The scalar line of Eq. (78) prints an uncontracted `partial_mu d`. Direct
   composition of Eq. (77), index consistency, and engineering dimension all
   require `d`.
2. The second row of the Eq. (80) node map repeats `Phi_2`, `Phi_3`, and
   `Phi_4`. They are `Phi_6`, `Phi_7`, and `Phi_8`.

The repository uses `epsilon_0123=-1` with a mostly-plus metric. The dual-H and
dual-F signs are translated explicitly into that convention. Antisymmetric
tensor pairs are stored once, so terms summed over both pair orders acquire the
corresponding exact factor of two.

## Reproduction

```bash
cargo test vector_tensor_4d --no-fail-fast
cargo run -- vector-tensor-4d-build
cargo run -- vector-tensor-4d-verify
```

Artifacts:

- `data/vector_tensor_4d.json`
- `results/vector_tensor_4d_validation.json`

## Boundary

This proves the sourced 4D component composition, including vector and
two-form gauge residues and every nonzero extension term. The separate
worldline calculation proves that the reduced residual is one central
generator. A final source-level bridge still has to match the normalized 4D
extension operator term by term to the central-coordinate transformations in
[arXiv:hep-th/9609016](https://arxiv.org/abs/hep-th/9609016), Eq. (4.6).
