# Eleven-Dimensional Gauge-Composition Build Order

## Objective

Determine which direct spinor-prepotential operators are well defined under
each of the six Lorentz-compatible first-derivative source transformations.

For a candidate operator

\[
A:\Psi_\alpha\longrightarrow H_\alpha{}^a
\]

and a source transformation

\[
G_p:\Lambda_{[p]}\longrightarrow\Psi_\alpha,
\]

the source-invariance condition is

\[
A\,G_p=0.
\]

The calculation treats the six parameter representations separately. Their
domains are inequivalent, so variations from different form degrees cannot
cancel one another.

## Completed input

The following inputs are complete:

- six exact \(C\Gamma^{[p]}\) intertwiners, \(0\leq p\leq5\);
- twelve exact leading \(D^{16}\Psi\to(10001)\) operators;
- forty-four exact first-momentum \(pD^{14}\Psi\to(10001)\) operators;
- the exact superspace anticommutator convention;
- the exact first-momentum joint artifacts.

The deterministic work list contains 336 gauge-operator pairs.

## Phase 1: zero-momentum source invariance

For each gauge degree \(p\) and each leading operator, construct

\[
D^{16}G_p\Lambda_{[p]}
\]

in exterior normal form at degree \(D^{17}\Lambda_{[p]}\).

There are

\[
6\times12=72
\]

jobs.

Each job records exact integer coordinates before any functional projection.
For each form degree, assemble the map from the twelve leading coefficients
to the complete \(D^{17}\) variation and compute its exact kernel.

The output is six subspaces

\[
K_p^{(0)}\subseteq\mathbb{Q}^{12}.
\]

For any selected set \(S\) of source gauge channels, the leading
source-invariant space is

\[
K_S^{(0)}=\bigcap_{p\in S}K_p^{(0)}.
\]

### Current status

The source-index convention has been fixed before first momentum. The
exterior engine stores \(D_\gamma\), so the written
\((C\Gamma^{[p]})_{\alpha\beta}D^\beta\) ansatz is executed by the
mixed-index operator \(\Gamma^{[p]}{}_\alpha{}^\gamma D_\gamma\).

All 72 corrected zero-momentum compositions are complete and deeply verified.
The exact ranks for form degrees zero through five are

\[
(1,11,11,12,12,11),
\]

with kernel dimensions

\[
(11,1,1,0,0,1).
\]

The degree-one, degree-two, and degree-five kernels are the same
scalar-factorizing line. The degree-zero kernel does not contain that line.
The degree-three and degree-four kernels are zero.

All 64 channel subsets have been classified. The empty subset has dimension
12, the degree-zero channel alone has dimension 11, and each nonempty subset
of degrees one, two, and five has dimension 1. The other 55 subsets have
dimension zero. The intersection of all six channels is zero.

The prior direct-\(C\Gamma^{[p]}\)-on-\(D_\gamma\) run is preserved as a
convention cross-check. It is not used for the source-quotient conclusion.

## Phase 2: first-momentum source invariance

### Current status

Not started. Phase 1 is complete. The first-momentum calculation must report
both the full solution dimension and whether any solution has a nonzero
leading component. In particular, degrees three and four already exclude a
nonzero leading component, but pure first-momentum solutions have not been
classified.

At order \(pD^{15}\Lambda\), include:

1. anticommutator terms from the twelve leading operators;
2. exterior-derivative terms from the forty-four first-momentum operators.

For each gauge degree, solve the exact compatibility system on all 56
operator coefficients, subject to the corresponding Phase 1 kernel.

This stage contains 336 gauge-operator jobs. As in the joint run, every job
must preserve:

- the exact raw residual stream;
- the fixed exact functional image;
- a manifest with source, executable, and fixture hashes;
- execution and resource logs;
- a transfer receipt.

## Phase 3: classify source gauge choices

The papers do not select the six channel coefficients. A nonzero coefficient
only selects a channel because its scale can be absorbed into its independent
parameter.

Classify all 64 channel subsets:

- no selected channel;
- each of the six individual channels;
- every nonempty intersection of their exact invariant spaces.

For each subset, report:

- leading invariant dimension;
- first-momentum completion dimension;
- whether any solution has a nonzero leading component;
- whether the scalar-factorizing direction survives.

Gauge-for-gauge reducibility may identify parameter descriptions, but it does
not change the condition \(A G_p=0\). It becomes necessary when the parameter
complex or quotient dimension is interpreted physically.

## Phase 4: retain or discard the hook

The completed joint calculation shows that no nonzero leading operator has a
vanishing hook residual after first-momentum correction. The conventional
eleven-dimensional constraints do not require the full `(11000)` hook to
vanish.

Therefore:

- if a nonzero source-invariant operator survives, retain its hook image as a
  candidate gauge-invariant torsion or curvature component and compute its
  next Bianchi map;
- if no nonzero source-invariant operator survives, a target gauge
  transformation is required before a quotient can be defined;
- do not quotient the hook residual by the six source maps alone.

## Phase 5: conditional target covariance

If an induced target transformation

\[
K_p:\Lambda_{[p]}\longrightarrow H_\alpha{}^a
\]

is supplied, replace source invariance by

\[
A\,G_p=K_p.
\]

This is a separate calculation. The cited sources do not print \(K_p\), so it
must not be inferred from the source intertwiners.

## Completion criterion

The gauge-composition stage is complete only when:

1. all 72 zero-momentum compositions are exact;
2. all 336 first-momentum compositions are exact;
3. all 64 source-channel subsets are classified;
4. every surviving operator is substituted into the complete raw residual;
5. source invariance is separated from conditional target covariance;
6. the hook is either retained with its Bianchi map or excluded by an
   explicitly stated constraint.
