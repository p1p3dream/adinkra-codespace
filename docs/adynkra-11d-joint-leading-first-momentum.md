# Eleven-Dimensional Joint Leading and First-Momentum Compatibility

## Question

The direct spinor-prepotential calculation has twelve leading maps

\[
D^{16}\Psi \longrightarrow (10001)
\]

and forty-four first-momentum corrections

\[
pD^{14}\Psi \longrightarrow (10001).
\]

The zero-momentum exterior derivative maps the twelve leading coefficients
into seven copies of the `(11000)` hook. Its exact matrix has rank seven and
nullity five. The next question is whether any vector in that
five-dimensional kernel can be completed by the forty-four first-momentum
corrections.

## Inputs

The calculation uses the committed exact certificates for:

- 12 level-16 leading couplings;
- 7 level-17 hook couplings;
- 28 level-14 source fixtures;
- 23 abstract level-14 source couplings;
- 44 embedded level-14 source maps;
- 4 vector times vector-spinor target couplings.

All source fixtures and all coupling coefficients are primitive integers.

## Reciprocal target maps

The four target certificates embed each intermediate irreducible in

\[
(10000)\mathbin{\otimes}(10001).
\]

The joint calculation also requires the reciprocal coupling

\[
(10000)\mathbin{\otimes}R\longrightarrow(10001)
\]

for each intermediate irreducible

\[
R=(00001),(01001),(10001),(20001).
\]

The Rust verifier constructs these four highest-weight systems from the
certified target embeddings. Each system must have a one-dimensional kernel
and zero raising residual before it can enter the joint matrix.

## First-momentum residual

The residual has normal-form degree \(pD^{15}\Psi\). Two contributions enter:

1. The anticommutator in the derivative of a leading term removes one
   spinor derivative and supplies one vector momentum:

   \[
   \{D_\alpha,D_\beta\}
   =2(\Gamma^a)_{\alpha\beta}p_a.
   \]

2. The exterior derivative of a first-momentum correction adds one spinor
   derivative while preserving its vector momentum.

Both contributions are projected through the same exact

\[
(00001)\mathbin{\otimes}(10001)\longrightarrow(11000)
\]

hook coupling.

The coordinate basis records:

- the vector-weight momentum index;
- the free spinor index;
- the sorted degree-15 exterior mask;
- separate real and imaginary integer coefficients.

The Clifford basis is converted to the same \(B_5\) vector-weight basis used
by the target intertwiners. The conversion preserves the stated
\(\{D,D\}=2\Gamma\mathbin{\cdot}p\) normalization.

## Exact solve

Let \(A_0\) be the completed \(7\times12\) zero-momentum hook matrix and let
\(A_1\) be the full \(pD^{15}\) coordinate matrix with 12 leading columns and
44 correction columns. The joint system is

\[
\begin{pmatrix}
A_0 & 0\\
A_1
\end{pmatrix}
\begin{pmatrix}
c_{\rm lead}\\
c_{\rm momentum}
\end{pmatrix}
=0.
\]

The full coordinate space is too large to retain as a dense row matrix. The
verifier therefore applies 512 fixed integer-valued functionals to every
full residual column and forms the exact rational normal matrix

\[
A_0^{T}A_0+A_1^{T}F^{T}FA_1.
\]

The functional matrix has rank no greater than the full coordinate matrix. A
nonzero \(56\times56\) minor in the functional image therefore proves that
the full 56-column system also has rank 56. This is an exact lower-rank
certificate, not a probabilistic equality assertion. If the functional image
does not have full rank, the command reports only the certified lower bound
and does not identify its kernel with the full coordinate kernel.

The report records:

- the exact functional-image rank and nullity;
- whether that image supplies a full-rank certificate for the complete
  coordinate system;
- a primitive integer basis for any functional-image kernel;
- direct substitution of every such vector into the functional matrix;
- the dimension of its projection onto the twelve leading coefficients;
- exact joint nullity when the functional image has full column rank;
- exclusion of leading extensions whenever the functional-image kernel has
  zero projection onto the twelve leading coefficients.

The second conclusion does not require the functional image to have full
rank. If \(B=LM\) is the functional image of the full coordinate matrix
\(M\), then

\[
\ker M\subseteq\ker B.
\]

Therefore, if every vector in \(\ker B\) has zero leading component, every
vector in the full coordinate kernel also has zero leading component.

## Reproduction

```bash
cargo run --release -- adynkra-11d-joint-compatibility
```

The command writes:

```text
results/adynkra_11d_joint_compatibility_matrix.json
```

## Distributed preservation contract

The distributed run must preserve each of the 56 columns independently. A
worker may not report a column as complete until it has written and synced:

- the exact sparse residual coordinate stream, before application of the
  fixed functionals;
- the exact functional image used in the rank calculation;
- the column label, ordinal, source and target labels, fixture copy, and
  conventions;
- the source revision and hashes of every fixture and executable input;
- start time, finish time, elapsed time, host identity, command line, peak
  resident memory, standard output, and standard error;
- SHA-256 hashes and byte counts for every artifact.

Workers write to a temporary directory and rename the directory atomically
only after all files and the per-column manifest are durable. Every completed
column is then copied immediately to a run-specific directory on stonkbot.
The worker verifies the stonkbot hashes against its local manifest before
continuing.

The source copy remains on the RunPod machine until:

1. stonkbot has acknowledged every file and hash;
2. a second copy has been synced to the local workstation;
3. the aggregate manifest confirms all expected column ordinals exactly once;
4. the merged functional matrix reproduces every per-column hash.

Failed and interrupted attempts retain their logs and partial files under a
separate incomplete directory. They are never accepted as completed columns
and are never allowed to overwrite a completed artifact. A RunPod machine is
not terminated until the complete manifest, artifact inventory, and final
matrix have been verified on both stonkbot and the local workstation.

## Boundary

This calculation tests the direct spinor-prepotential compatibility
conditions through first momentum order. It does not impose the six possible
gauge-parameter maps, construct a curvature, supply an action, or derive a
field equation.

## Result

The distributed production run completed all 56 columns:

- 12 leading columns;
- 44 first-momentum correction columns;
- 2,405,151,800 nonzero exact residual records;
- 91,395,769,520 uncompressed artifact bytes;
- 21,489,272,247 compressed artifact bytes.

Every compressed stream, uncompressed stream, functional file, manifest,
execution log, and transfer receipt passed its SHA-256 verification.

The preserved run identifier is `joint-20260724T030424Z`. The final version-2
report has SHA-256
`521e0a3dfbe85787abc738302801d8ce5ce2ee5f58f56c146844251e66ded070`.

The 519-row exact functional image has rank 51 and nullity 5. All five
functional-kernel basis vectors have zero entries in the twelve leading
coordinates. Consequently, every null vector of the full coordinate system
also has zero leading component. None of the five zero-momentum hook-kernel
directions can be completed by the 44 first-momentum corrections.

This conclusion is exact even though the full coordinate nullity is not yet
fixed. The remaining uncertainty is confined to the five correction-only
functional-kernel directions. The full coordinate rank lies between 51 and
56, and its nullity lies between 0 and 5. Determining which correction-only
directions, if any, vanish in the full residual does not alter the exclusion
of a nonzero leading extension.

This is a negative result for the direct spinor-prepotential calculation
through first momentum order. It does not test the six possible
gauge-parameter maps or the other cases listed in the boundary above.
