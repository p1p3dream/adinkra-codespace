# Eleven-Dimensional Spinor-Prepotential Gauge Intertwiners

## Question

The proposed fundamental prepotential is an unconstrained spinor superfield
\(\Psi_\alpha\), related to the scalar semi-prepotential by

\[
V=D^\alpha\Psi_\alpha.
\]

The cited proposal does not give a gauge transformation for \(\Psi_\alpha\).
The complete Lorentz-compatible first-derivative ansatz is

\[
\delta\Psi_\alpha
=
\sum_{p=0}^{5}c_p
\left(C\Gamma^{a_1\ldots a_p}\right)_{\alpha\beta}
D^\beta\Lambda_{a_1\ldots a_p}.
\]

The six parameter representations are

\[
(00000),(10000),(01000),(00100),(00010),(00002),
\]

with dimensions

\[
1,11,55,165,330,462.
\]

Their dimensions sum to \(1024=32^2\). Each representation occurs once in
the spinor square.

## Current status

The six source intertwiners and their old source-invariance screens are
complete. Since those screens were built, the program has also completed:

- the exact target-resolved `11x32` vector-spinor stream;
- the 11D Majorana target join and free light-cone supersymmetry maps;
- all 42 level-18 source kernels and all 77 embedded source-target maps;
- a convention-fixed physical partial-`F_X` bounded screen over all six form
  degrees and 56 recorded operators.

The physical bounded screen has exact global rank 49 and nullity zero, so it
excludes the recorded five-dimensional leading `F_X` kernel plus 44
first-momentum correction directions. Complete physical `K`, full `F`, the
physical routing and quotient on the 77 blocks, generic-polynomial FAG, and
covariant off-shell closure remain unavailable.

## Exact construction

The Rust verifier constructs every matrix

\[
C\Gamma^{[p]},\qquad 0\leq p\leq5,
\]

in the same exact Gaussian-rational Clifford basis used by the
eleven-dimensional bridge calculation.

It checks:

- all 1,024 form matrices;
- one nonzero unit coefficient in every row and column of every matrix;
- rank 32 for every component matrix;
- the required symmetric or antisymmetric spinor-index symmetry;
- all 524,800 diagonal and independent off-diagonal Hermitian inner products;
- norm 32 for every matrix;
- zero pairwise inner product between distinct matrices.

The 1,024 matrices therefore form an exact basis of the full spinor-square
operator space.

## Scalar semi-prepotential

For the inherited relation \(V=D^\alpha\Psi_\alpha\):

- the one-form, two-form, and five-form channels give zero scalar variation
  at zero momentum;
- only the two-form and five-form channels give zero scalar variation at
  generic momentum.

This reproduces the earlier Clifford audit. It does not establish that the
two-form or five-form channels are physical gauge symmetries of the
fundamental prepotential.

## Quotient conditions

The six source intertwiners do not by themselves define a quotient of the
vector-spinor target or the hook residual.

For an operator \(A:\Psi\to H\) to be defined on the source quotient, it must
satisfy

\[
A\,G_p=0
\]

for every selected source gauge channel \(G_p\).

A weaker covariant condition,

\[
A\,G_p=K_p,
\]

requires an independently specified target gauge transformation \(K_p\).
Neither cited source prints such a transformation for the proposed
vector-spinor target. Therefore the equation

\[
Mx\in\operatorname{Im}G
\]

is not justified if \(G\) denotes only the six source maps.

The zero-momentum compositions and the first-momentum source exclusion screens
are complete. No nonzero leading operator descends through first momentum
under any nonempty selection of the six candidate source channels. This is a
statement about source invariance `A G_p=0`. Cases requiring a nonzero target
transformation remain conditional until that transformation is supplied.

The target-resolved API fixes a deterministic work list of 408 source
gauge-operator compositions:

- 72 zero-momentum \(D^{17}\) jobs from six gauge channels and twelve leading
  operators;
- 336 first-momentum \(pD^{15}\) jobs from six gauge channels and all 56
  operators.

The exact `11x32` target stream interface is complete, and all 77 level-18
embedded maps are complete. The typed target-quotient APIs also exist. The
physical channel-to-block routing and physical coefficients do not, so the
actual target gauge quotient remains false.

The completed source calculation excludes a nonzero leading solution whose
hook residual vanishes. The source-invariance screens also exclude a nonzero
leading completion for every nonempty source-channel selection. They do not
test an induced target gauge transformation or the route that retains the hook
as a torsion or curvature component.

## Index convention

The written ansatz contains \(D^\beta\). The exterior engine stores
lower-index derivative components \(D_\gamma\). Therefore the executed map is

\[
\delta\Psi_\alpha
=
\sum_{p=0}^{5}c_p
\left(\Gamma^{[p]}\right)_\alpha{}^\gamma
D_\gamma\Lambda_{[p]},
\]

which is equivalent to
\((C\Gamma^{[p]})_{\alpha\beta}D^\beta\Lambda_{[p]}\).
In particular, the degree-zero operator is the identity.

An earlier run applied the lowered bilinear \(C\Gamma^{[p]}\) directly to
the lower-index derivative basis. That run is retained as a convention
cross-check, but it does not execute the written mixed-index map and is not
used for the source-quotient conclusion.

## Zero-momentum composition result

All 72 corrected mixed-index compositions are complete. Every compressed and
uncompressed residual stream passed its recorded SHA-256 check. The six exact
maps on the twelve leading operators have the following ranks and kernel
dimensions:

| Form degree | Dynkin label | Parameter dimension | Rank | Kernel dimension |
|---:|:---:|---:|---:|---:|
| 0 | `00000` | 1 | 1 | 11 |
| 1 | `10000` | 11 | 11 | 1 |
| 2 | `01000` | 55 | 11 | 1 |
| 3 | `00100` | 165 | 12 | 0 |
| 4 | `00010` | 330 | 12 | 0 |
| 5 | `00002` | 462 | 11 | 1 |

The degree-one, degree-two, and degree-five kernels are the same line. Its
primitive coordinates in the ordered leading basis are

```text
(70560, 10080, -3780, -15120, 7560, -498960,
 -5040, -35, 63, 0, -120, 0).
```

This is the scalar-factorizing leading direction found independently in the
level-17 derivative calculation. The degree-zero kernel does not contain this
line. Degrees three and four admit no nonzero source-invariant leading
operator.

The exact classification of all 64 channel subsets gives:

- the empty subset has dimension 12;
- the degree-zero channel alone has dimension 11;
- each nonempty subset of degrees one, two, and five has the same
  one-dimensional scalar-factorizing kernel;
- the other 55 subsets have dimension zero.

Thus nine of the 64 subsets have a nonzero leading space, including the empty
subset. The intersection of all six channels is zero. These are
source-invariance statements for the six candidate channels. The cited papers
do not select a physical channel subset.

The degree-zero channel does not leave \(V=D^\alpha\Psi_\alpha\) invariant.
It remains relevant only if \(V\) is allowed an induced transformation or is
not required to define the source quotient. The degree-two and degree-five
channels leave the scalar-factorizing leading direction invariant at zero
momentum, but that direction does not survive their first-momentum screens.

## Historical first-momentum source exclusion result

Degrees three and four need no first-momentum calculation because their
zero-momentum leading kernels vanish. Degrees zero, one, two, and five were
combined with all forty-four first-momentum correction columns.

| Form degree | Parameter components evaluated | Functional rank | Functional nullity | Leading projection rank |
|---:|:---:|---:|---:|---:|
| 0 | all, component 0 of 1 | 17 | 38 | 0 |
| 1 | component 0 of 11 | 42 | 3 | 0 |
| 2 | component 0 of 55 | 33 | 12 | 0 |
| 5 | component 0 of 462 | 41 | 4 | 0 |

This table records the original one-component functional screens. The later
source-only merge evaluated every parameter component for degrees one, two,
and five. Each complete merge has functional rank 42, nullity 3, zero leading
projection rank, and exact zero residual on its functional kernel. Their
artifact SHA-256 values are:

```text
form 1: f183def003a71cd08b7516ad5a666e589eff20629706bdda64bb5d0eb4e3b62c
form 2: 9177fe087728bced2df21a984020a1d7d5c485a59e01f9ac1094673ccc32a7cd
form 5: 281999a56b85ab59b7fa50a40c4b2f6afa645f4c2cb24fc6563d60c621b272c2
```

The screens use exact integer linear functionals of the first-momentum source
term stream. The zero leading projection in each functional kernel excludes a
nonzero leading completion. The reported functional nullities do not classify
pure first-momentum solutions.

Together with the zero-momentum results, every nonempty subset of candidate
source channels is excluded from carrying a nonzero leading operator through
first momentum in the recorded source ansatz. This is a result about the
source-invariance equation \(A G_p=0\). It does not exclude target covariance
\(A G_p=K_p\), and it does not decide the retained-hook and Bianchi route.

## Physical partial-F_X bounded screen

The physical `F_X` calculation is distinct from the source screens above. It
uses the frozen convention-fixed physical-curvature v10 input snapshot and the
exact target-basis join, then composes the implemented `X_[2]` and `X_[5]`
sectors with one declared parameter component and target basis ordinal 319.
The current enriched physical-curvature envelope separately records and
validates the frozen input, the physical `F_X` report, and the checkpoint
promotion provenance.

All six form degrees and all 56 recorded operators were processed, producing
336 complete operator checkpoints. The exact joint functional results are:

| Form degree | Selected parameter | Selected target | Operators | Joint rank | Joint nullity |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 319 | 56 | 11 | 38 |
| 1 | 0 | 319 | 56 | 46 | 3 |
| 2 | 0 | 319 | 56 | 35 | 14 |
| 3 | 0 | 319 | 56 | 49 | 0 |
| 4 | 0 | 319 | 56 | 48 | 1 |
| 5 | 0 | 319 | 56 | 45 | 4 |

The stacked all-six system has exact rank 49 and nullity zero in the recorded
five-leading-kernel-plus-forty-four-first-momentum coefficient space. It
therefore excludes every nonzero vector in that bounded 49-dimensional space.

This is not a complete physical FAG calculation. Parameter and target
projections are incomplete, `F_X` omits `J` and `W`, the physical source map
`K` has not been selected, physical routing and quotient coefficients on the
77 target blocks are missing, and higher momentum descendants are not
exhausted. Accordingly `generic_k_solved`,
`all_six_physical_fag_channels_checked`, and `physical_fag_established` all
remain false.

Pinned SHA-256 values:

```text
frozen F_X input v10:     c308ed82072b835776aa4451751434e500daab922926d12a0dc67735c923083f
current physical envelope: 3c31f29d0853f415a11adda78bbb52368e59d848013486affeb4aa9e88a23b13
physical F_X screen:       5a9a6e13ff57789817689a6d1791ec3d4e94b5731af02a1ed618bedd1a30f4f9
checkpoint promotion:      98941c4cfa46462d519bbe823489622bbad56cc7a6bb3a01596cc3fdf6b8aec4
K/FAG harness:             11ec33c36d9536e17e617839cc8dbabc885b9d30bf13ff05a4d0dc5e6b9fe562
```

The promotion manifest pins all 336 checkpoint hashes. It reports 164
existing files verified byte-for-byte, 172 missing files copied from the
completed candidate corpus, and zero partial replacements.

The physical `F_X` report remains pinned to the immutable `c308...` input
snapshot. The `3c31...` current envelope is a separate enriched status
artifact that validates this provenance. It is not a replacement input and
does not establish complete `F`.

Corrected mixed-index run:

```text
gauge-mixed-index-20260724T133344Z
```

Compact result SHA-256 values:

```text
form 0: 184c3a683e47e4c8ec3763e1318fb2d200fe3f1c44e2d4eae6273da317064d40
form 1: b2cdb3733d4ca6eb156ba70a399fa0d4a05be8704d65e843ca97627c1c9babf9
form 2: 606f919a164049ffcd7bed03afb791abf3644c1395172bad45b25ab0500d8807
form 3: c9f5ddf4372e43519e480ef6ddb986b347a208859171372e445b813f9c21479e
form 4: 1ca023a678261b648deed824f3c6c3249a8ba96f0deb0dc98574fd23a6e5f841
form 5: a5bc727926fe9509aa72858fff151b15a6f26ef36f32cd403d828b7b8d31ddaa
subsets: 4f35f3f5efc6b59cfc30961247efde540c1950d562e061115e5073dad1898fba
```

## Reproduction

```bash
cargo run --release -- adynkra-11d-gauge-intertwiner-verify
cargo run --release -- adynkra-11d-gauge-zero-column 0 0 RUN_ROOT
cargo run --release -- adynkra-11d-gauge-zero-merge 0 RUN_ROOT --deep
cargo run --release -- adynkra-11d-gauge-zero-classify RUN_ROOT
cargo run --release -- adynkra-11d-gauge-first-functional-stream-prefix 1 0 1 RUN_ROOT
cargo run --release -- adynkra-11d-gauge-first-functional-merge 1 RUN_ROOT ZERO_ROOT
cargo test --release eleven_dimensional_gauge
cargo test --release eleven_dimensional_physical_curvature
cargo test --release eleven_dimensional_k_fag_solver
```

Output:

```text
results/adynkra_11d_gauge_intertwiners.json
results/adynkra_11d_gauge_zero_momentum_form_0.json
...
results/adynkra_11d_gauge_zero_momentum_form_5.json
results/adynkra_11d_gauge_zero_momentum_subsets.json
results/adynkra_11d_gauge_first_momentum_functional_form_0.json
results/adynkra_11d_gauge_first_momentum_functional_form_1.json
results/adynkra_11d_gauge_first_momentum_functional_form_2.json
results/adynkra_11d_gauge_first_momentum_functional_form_5.json
results/adynkra_11d_first_momentum_gauge_functional_p1.json
results/adynkra_11d_first_momentum_gauge_functional_p2.json
results/adynkra_11d_first_momentum_gauge_functional_p5.json
results/adynkra_11d_target_stream_validation.json
results/adynkra_11d_b5_majorana_target_join.json
results/adynkra_11d_level18_embedded_maps.json
results/adynkra_11d_level18_target_quotient_basis.json
results/adynkra_11d_first_momentum_physical_fx_functional.json
results/adynkra_11d_first_momentum_physical_fx_checkpoint_promotion.json
results/adynkra_11d_k_fag_polynomial_harness.json
```

SHA-256:

```text
6a9f5e3b7d4102eb54ccfb33c384da359ce918c16c551f13c0bb04170030d90f
```

Implementation:

- `src/eleven_dimensional_gauge.rs`
- `src/eleven_dimensional_clifford.rs`
- `src/eleven_dimensional_level16_couplings.rs`
- `src/eleven_dimensional_level18_embedded.rs`
- `src/eleven_dimensional_level18_target_quotient.rs`
- `src/eleven_dimensional_b5_majorana_target_join.rs`
- `src/eleven_dimensional_physical_curvature.rs`
- `src/eleven_dimensional_k_fag_solver.rs`

## Result

All six candidate first-derivative source intertwiners are constructed and
pass the exact algebraic checks. The target-resolved stream, Majorana join,
level-18 maps, and physical partial-`F_X` bounded screen are also complete.
The rank-49 screen rules out the recorded `5+44` coefficient space on its
declared slice.

This remains a complete Lorentz-compatible source ansatz, not a selection of
a physical gauge law. Complete `H_hat -> F`, physical `K`, physical routing
and target quotient, generic-polynomial FAG, and covariant off-shell closure
remain false.

## References

1. S. J. Gates Jr., Y. Hu, and S.-N. H. Mak, *Adinkra Foundation of
   Component Decomposition and the Scan for Superconformal Multiplets in 11D,
   N = 1 Superspace*, arXiv:2002.08502, Added Note in Proof, Eq. (6.3).
2. S. J. Gates Jr., Y. Hu, and S.-N. H. Mak, *Weyl Covariance, and Proposals
   for Superconformal Prepotentials in 10D Superspaces*, arXiv:2007.05097.

The second reference is a 10D Weyl-covariance and prepotential paper. It is
used for conventions and structural motivation only. It is not an 11D
spinorial-cohomology computation, does not select the physical 11D gauge map
`K`, and does not establish 11D off-shell closure.
