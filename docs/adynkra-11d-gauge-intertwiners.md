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

The next exact calculation is to compose each \(G_p\) with the twelve leading
and forty-four first-momentum operators. This will determine which candidate
operators descend to a source quotient. Cases requiring a nonzero target
transformation must remain conditional until that transformation is supplied.

The verifier fixes a deterministic work list of 336 gauge-operator
compositions:

- 72 zero-momentum \(D^{17}\) jobs from six gauge channels and twelve leading
  operators;
- 336 first-momentum \(pD^{15}\) jobs from six gauge channels and all 56
  operators.

The completed joint calculation already excludes a nonzero leading solution
whose hook residual vanishes. Adding source-gauge invariance can only restrict
that solution space. It can reopen the route only if the target has its own
gauge equivalence or if the nonzero hook is retained as an allowed torsion or
curvature component.

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
momentum. First-momentum compatibility remains to be computed.

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
cargo test --release eleven_dimensional_gauge
```

Output:

```text
results/adynkra_11d_gauge_intertwiners.json
results/adynkra_11d_gauge_zero_momentum_form_0.json
...
results/adynkra_11d_gauge_zero_momentum_form_5.json
results/adynkra_11d_gauge_zero_momentum_subsets.json
```

SHA-256:

```text
6a9f5e3b7d4102eb54ccfb33c384da359ce918c16c551f13c0bb04170030d90f
```

Implementation:

- `src/eleven_dimensional_gauge.rs`
- `src/eleven_dimensional_clifford.rs`

## Result

All six candidate first-derivative intertwiners are constructed and pass the
exact algebraic checks. This is a complete Lorentz-compatible ansatz, not a
selection of a physical gauge law.

## References

1. S. J. Gates Jr., Y. Hu, and S.-N. H. Mak, *Adinkra Foundation of
   Component Decomposition and the Scan for Superconformal Multiplets in 11D,
   N = 1 Superspace*, arXiv:2002.08502, Added Note in Proof, Eq. (6.3).
2. S. J. Gates Jr., Y. Hu, and S.-N. H. Mak, *Weyl Covariance, and Proposals
   for Superconformal Prepotentials in 10D Superspaces*, arXiv:2007.05097.
