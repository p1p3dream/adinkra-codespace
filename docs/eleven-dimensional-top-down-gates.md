# Eleven-dimensional top-down gates

## Scope

This layer attacks the 11D problem from the target theory downward. It does not
enumerate N=32 worldline graphs. It builds exact sparse gauge, curvature,
Bianchi, equation, and representation-theoretic complexes, then asks which
superfield constructions can map into them.

The current result is a collection of exact bounded gates. It is not an
irreducible off-shell 11D supermultiplet and not a supersymmetric extension of
Einstein's equation.

## Gate 1: free component complex and physical real form

`eleven_dimensional_free_complex` constructs exact sparse matrices over
Gaussian rationals at null momentum `p=(1,1,0,...,0)` in mostly-plus signature.

| Sector | Potential dimension | Gauge rank | Equation rank | Physical quotient |
| --- | ---: | ---: | ---: | ---: |
| graviton `h_ab` | 66 | 11 | 11 | 44 |
| three-form `A_abc` | 165 | 45 | 36 | 84 |
| gravitino `psi_a` | 352 | 32 | 192 | 128 |

Every implemented gauge-to-curvature, curvature-to-Bianchi,
gauge-to-Euler-Lagrange, and Euler-Lagrange-to-Noether composition vanishes
exactly at three checked momenta. The three-form sector includes its full
scalar-to-one-form-to-two-form reducibility chain and both `d^2=0` gates.

The later target-equation and B5-to-Majorana joins now go beyond the original
complexified census:

- the 11D Majorana real form is certified;
- the physical `44+84|128` null-momentum census is exact;
- light-cone supersymmetry maps between the sectors are certified;
- the light-cone supersymmetry closure residual is zero;
- the 320-dimensional gamma-traceless vector-spinor target has an exact
  two-sided B5-to-Cartesian-Majorana basis join.

This is an on-shell light-cone closure result for the free target. Covariant
off-shell superfield closure remains false.

## Gate 2: hook continuation and level-18 maps

The committed zero-momentum level-16 incidence differential has shape `7x12`,
rank 7, nullity 5, and left nullity zero. Consequently every relaxed rational
next-Bianchi row `B` satisfying

```text
B d16 = 0
```

is forced to vanish, and the bounded zero-momentum group
`H17=ker(d17)/im(d16)` is zero.

For the hook tensor `(11000)`, spinor tensoring produces four multiplicity-one
targets:

| Target | Dimension |
| --- | ---: |
| `(01001)` | 1,408 |
| `(10001)` | 320 |
| `(11001)` | 10,240 |
| `(20001)` | 1,760 |

The former level-18 worklist is now complete. Exact kernels exist for all 42
required source copies across 16 irreps, and all 77 embedded source-target maps
have zero residual. The typed 77-block target basis and exact rank, kernel,
image-containment, and quotient APIs are also present.

What remains missing is physical rather than combinatorial. The 77 blocks do
not yet have source-selected physical channel routing or physical coefficients,
so the actual target gauge quotient has not been computed. The zero-momentum
hook result therefore does not by itself prove momentum-dependent,
gauge-quotiented superspace cohomology.

## Gate 3: target-resolved stream

The exact target-resolved composition interface is implemented. It retains the
full `11x32` ambient vector-spinor coordinates, projects to the deterministic
320-state `(10001)` target basis, and supports both

```text
D^17 Lambda
p D^15 Lambda
```

source compositions with exact Gaussian-rational coefficients. The source-side
inventory consists of 12 leading operators and 44 recorded first-momentum
corrections across six inequivalent gauge-form channels.

This finishes the target stream API. It does not select the physical source
map `K`, provide physical routing through the 77 level-18 blocks, or exhaust
all parameter components and target coordinates.

## Gate 4: physical partial-F_X bounded screen

A frozen convention-fixed physical-curvature v10 input snapshot supplies the
implemented `X_[2]` and `X_[5]` sectors of `F`. The exact first-momentum run
composed one declared parameter component and target basis ordinal 319 for
every pair of six gauge degrees and 56 recorded operators. This produced 336
complete operator checkpoints. The current enriched physical-curvature
envelope separately records and validates the frozen input, the physical
partial-`F_X` report, and the checkpoint promotion provenance.

The coefficient domain is the five-dimensional leading `F_X` kernel plus 44
recorded first-momentum correction directions, for 49 variables total. The
all-six bounded screen gives

```text
global X_[2] rank/nullity: 49/0
global X_[5] rank/nullity: 49/0
global joint rank/nullity: 49/0
```

Dimension saturation is exact. Therefore the declared physical partial-`F_X`
slice excludes every nonzero vector in the recorded `5+44` coefficient space.
This is a real negative control for that finite space.

It is not the full identity `F A G_p = 0`. The run selects only parameter
component 0 and target basis ordinal 319, `F_X` omits the required `J` and `W`
sectors, and the `12+44` operator ansatz omits higher momentum descendants.
Thus complete parameter coverage, complete target coverage, generic
polynomial `K`, all-six physical FAG coverage, and physical FAG establishment
all remain false.

### Pinned provenance

- frozen physical-curvature v10 `F_X` input snapshot SHA-256:
  `c308ed82072b835776aa4451751434e500daab922926d12a0dc67735c923083f`
- current enriched physical-curvature v10 envelope SHA-256:
  `3c31f29d0853f415a11adda78bbb52368e59d848013486affeb4aa9e88a23b13`
- physical partial-`F_X` artifact SHA-256:
  `5a9a6e13ff57789817689a6d1791ec3d4e94b5731af02a1ed618bedd1a30f4f9`
- 336-checkpoint promotion manifest SHA-256:
  `98941c4cfa46462d519bbe823489622bbad56cc7a6bb3a01596cc3fdf6b8aec4`
- K/FAG bounded-control harness SHA-256:
  `11ec33c36d9536e17e617839cc8dbabc885b9d30bf13ff05a4d0dc5e6b9fe562`

The promotion manifest contains exactly 336 checkpoint hashes. It records 164
existing checkpoints verified byte-for-byte, 172 missing checkpoints copied
from the completed candidate corpus, and zero partial replacements.

The physical partial-`F_X` report is pinned to the immutable `c308...` input
snapshot. The `3c31...` current envelope is a distinct enriched status
artifact. It validates that input relationship without replacing the input
hash or claiming complete `F`.

## Source screens versus the physical screen

The older zero-momentum and first-momentum source screens test equations of the
form

```text
A G_p = 0
```

before applying the convention-fixed physical curvature operator. They are
useful source-invariance exclusions, but they are not physical FAG tests.

The new 336-checkpoint result applies the implemented physical `F_X` consumer
to a declared target-resolved slice. It is therefore physically closer, but it
is still only partial `F_X`, not complete `F`, and it still lacks physical `K`
and the target gauge quotient. The two result families must not be merged into
a claim of full gauge invariance.

## Aggregate status

Green gates:

1. exact free graviton, three-form, and gravitino complexes
2. Majorana real form and free light-cone `44+84|128` supersymmetry closure
3. target-resolved exact `11x32` composition stream
4. all 42 level-18 source kernels and all 77 embedded maps
5. typed target-quotient APIs and synthetic quotient controls
6. convention-fixed physical `F_X` bounded screen with exact rank 49 and
   nullity zero

Claims that remain false:

1. a source-selected physical `Psi -> H_hat` map `K`
2. complete `H_hat -> F`, including induced `J`, `T`, and `W` on the physical
   quotient
3. physical channel routing and coefficients on the 77 target blocks
4. the actual physical target gauge quotient
5. generic-polynomial `F A G_p = 0` for all parameters and targets
6. momentum-dependent superspace cohomology identifying an irreducible
   off-shell multiplet
7. covariant off-shell superfield closure and a linearized 11D Adynkra
   equation
8. agreement with nonlinear 11D component equations

## Next executable steps

1. Complete the convention-fixed `H_hat -> J,T,W,X -> F` operator on the
   derivative-Lorentz quotient.
2. Select or derive physical `K`, then route each physical channel through the
   77 embedded target blocks.
3. Compute the actual target gauge quotient with physical coefficients.
4. Extend the bounded screen to every parameter component, target coordinate,
   required lower symbol, and momentum descendant under a proved degree bound.
5. Run the generic polynomial identity independently in all six physical
   channels.
6. Only after those gates pass, construct and test covariant off-shell
   superfield closure and a linearized equation.

## Reference roles

- arXiv:2002.08502 supplies the 11D component scan and the conjectured
  prepotential context. It does not construct the physical map `K` or full
  curvature `F`.
- arXiv:2007.05097 studies Weyl covariance and proposed prepotentials in 10D.
  It is useful for conventions and structural motivation, but it is not an 11D
  spinorial-cohomology computation and not an oracle for 11D off-shell closure.
- Howe's 11D superspace analysis, Cederwall's pure-spinor work, and explicit
  linearized 11D component treatments are the appropriate comparison set for
  torsion constraints, on-shell cohomology, and physical component content.

## Reproduction

```bash
cargo test --release eleven_dimensional_free_complex
cargo test --release eleven_dimensional_target_equation_complex
cargo test --release eleven_dimensional_b5_majorana_target_join
cargo test --release eleven_dimensional_level18_embedded
cargo test --release eleven_dimensional_level18_target_quotient
cargo test --release eleven_dimensional_physical_curvature
cargo test --release eleven_dimensional_k_fag_solver
cargo test --release eleven_dimensional_top_down
```

Key artifacts:

- `results/adynkra_11d_free_complex_validation.json`
- `results/adynkra_11d_target_equation_complex.json`
- `results/adynkra_11d_b5_majorana_target_join.json`
- `results/adynkra_11d_target_stream_validation.json`
- `results/adynkra_11d_level18_embedded_maps.json`
- `results/adynkra_11d_level18_target_quotient_basis.json`
- `results/adynkra_11d_physical_curvature_validation.json`
- `results/adynkra_11d_first_momentum_physical_fx_functional.json`
- `results/adynkra_11d_first_momentum_physical_fx_checkpoint_promotion.json`
- `results/adynkra_11d_k_fag_polynomial_harness.json`
