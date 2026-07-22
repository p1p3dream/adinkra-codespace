# Four-dimensional N=1 rank-two Lorentz intertwiners

## Result

The Rust implementation constructs explicit irreducible projectors for the
rank-two tensor in Eqs. (2.5) and (2.18) of Gates and Hu,
arXiv:2407.09334v1:

```text
[1,1] tensor [1,1] = [0,0] + [2,2] + [0,2] + [2,0]
```

The four projectors isolate:

| Sector | Dynkin label | Rank |
|---|---|---:|
| trace | `[0,0]` | 1 |
| symmetric traceless | `[2,2]` | 9 |
| self-dual 2-form | `[0,2]` | 3 |
| anti-self-dual 2-form | `[2,0]` | 3 |

The duality-label convention follows Eq. (2.6) of the paper.

All four projectors are idempotent. All 12 ordered cross-products vanish. Their
ranks sum to 16, and their sum is the identity on the complete rank-two tensor
space. Reconstruction of all 16 basis tensors has zero residual.

The implementation also constructs the six `so(4)` vector generators and their
induced action on the tensor product. All 24 generator-projector commutators
vanish exactly, establishing equivariance in the chosen complexified `SO(4)`
basis.

## Vector-spinor sectors

The companion implementation reproduces the two decompositions in Eqs.
(2.13), (2.14), and (2.19):

```text
[1,1] tensor [1,0] = [0,1] + [2,1]
[1,1] tensor [0,1] = [1,0] + [1,2]
```

For each chirality, complementary exchange projectors isolate the rank-2
spinor trace and rank-6 spin-three-halves sector. Both pairs are complete,
orthogonal, idempotent, and equivariant under the six generators of
`sl(2)_L + sl(2)_R`. All 24 generator-projector commutators vanish exactly.

## Artifacts

- `src/lorentz_intertwiners.rs`: exact rational projectors and checks
- `results/adynkra_4d_n1_intertwiner_validation.json`: validation report
- `src/vector_spinor_intertwiners.rs`: exact chiral vector-spinor projectors
- `results/adynkra_4d_n1_vector_spinor_validation.json`: validation report

## Reproduction

```bash
cargo run --release -- adynkra-intertwiner-verify \
  > results/adynkra_4d_n1_intertwiner_validation.json
cargo run --release -- adynkra-vector-spinor-verify \
  > results/adynkra_4d_n1_vector_spinor_validation.json
cargo test lorentz_intertwiners
cargo test vector_spinor_intertwiners
```

## Boundary

This supplies explicit irreducible intertwiners for the bosonic rank-two and
both vector-spinor sectors. The fundamental derivative intertwiners and all
repeated sectors in the six published genomes are documented separately in
[`adynkra-4d-n1-derivative-intertwiners.md`](adynkra-4d-n1-derivative-intertwiners.md).
Assembly of the prepotential gauge map and its cohomology remain open.

## Reference

S. J. Gates Jr. and Y. Hu, "Adynkra Genomes, Adynkrafields, and the 4D, N=1
Supergravity Superfield Prepotential," [arXiv:2407.09334v1](https://arxiv.org/abs/2407.09334).
