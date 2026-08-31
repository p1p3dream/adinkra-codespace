# GPU contract for eleven-dimensional local-Lorentz and target-gauge descent

Date: 2026-08-30

Status: implementation contract, not a completed descent certificate

## 1. Purpose and claim boundary

This document turns the local-Lorentz and target-gauge parts of the
irreducibility roadmap into exact, GPU-checkable jobs. It separates three
questions that must not be conflated:

1. Does the raw constrained-frame construction descend from a chosen frame
   lift to the gamma-traceless superfield \(\widehat H\)?
2. Does a supplied target-gauge map \(K\) satisfy \(F K=0\) exactly?
3. Is the supplied \(K\) the complete local polynomial target-gauge module?

The first two are exhaustive composition checks and are realistic targets for
a wall time below 30 minutes on the RTX 4090 after the adapters described here
exist. The third is a module-discovery and completeness proof. GPU rank and
Macaulay kernels can accelerate it, but no 30-minute completeness claim is
valid before representation blocks and a regularity or degree bound are
proved.

The current physical operator is also conditional. The direct Riemann and
direct gravitino-curl branches are exact. The closed Eq. (40)
\(\Psi_{[3]}\) branch is not yet authoritatively identified and normalized as
the physical \(A_3/G_4\) branch. No final local-Lorentz or target-gauge
certificate may bind an operator that still carries that conditional tag.

## 2. Rings, bases, and typed maps

Let

\[
k=\mathbb Q(i),\qquad R=k[p_0,\ldots,p_{10}].
\]

The ordered-superderivative algebra is

\[
\mathcal A=
R\langle D_0,\ldots,D_{31}\rangle/
\left(D_\alpha D_\beta+D_\beta D_\alpha
-i(C\Gamma^a)_{\alpha\beta}p_a\right).
\]

Its canonical PBW keys are \((I,\nu)\), where \(I\) is a 32-bit exterior
spinor mask stored in descending \(D\) order and
\(\nu\in\mathbb N^{11}\) is a momentum exponent vector. The executable
normal form is in
`src/eleven_dimensional_superderivative_normal_form.rs`.

The relevant finite Cartesian fibers are:

| symbol | type | dimension | repository basis |
|---|---|---:|---|
| \(S\) | Majorana spinor | 32 | Cartesian Majorana |
| \(V\) | Lorentz vector | 11 | mostly-plus Cartesian |
| \(\Lambda^2V\) | local-Lorentz two-form | 55 | increasing-mask order |
| \(\widehat H\) | gamma-traceless vector-spinor `(10001)` | 320 | ten spatial vector slots times 32 spinors, with the time slot solved by the gamma trace |
| \(\Sigma\) | scale canary | 1 | separate final source column |

The complete-F production source is therefore
\(\widehat H\oplus\Sigma\), of dimension 321. Target-gauge calculations must
select source ordinals 0 through 319. Ordinal 320 is the scale canary and is
not in the codomain of physical \(K\).

### 2.1 Source and target maps

The four maps in the main roadmap retain their distinct types:

\[
G_q:\Lambda_{[q]}\longrightarrow\Psi_{\mathrm{source}},
\qquad q=0,\ldots,5,
\]

\[
A_c:\Psi_{\mathrm{source}}\longrightarrow\widehat H,
\]

\[
K:\Xi_{\mathrm{target}}\longrightarrow\widehat H,
\]

\[
F:\widehat H\longrightarrow\mathcal E_{\mathrm{phys}}.
\]

The six source-gauge domains have dimensions
\(1,11,55,165,330,462\), totaling 1,024. They are not six scalar
coefficients of \(K\). The 77 incidence blocks in the existing quotient
backend form a separate 439,904-dimensional direct sum. They do not become a
physical target quotient until an exact basis join and convention-fixed
routing from \(K\) are supplied.

### 2.2 Local-Lorentz lift maps

Let \(\widetilde{\mathcal H}\) be the raw constrained-frame input containing
\(H_\alpha{}^a\), the scale, and \(\Psi_{[2]}\). Define

\[
q:\widetilde{\mathcal H}\longrightarrow\widehat H\oplus\Sigma
\]

to remove the gamma trace and the pure local-Lorentz two-form, and let

\[
s:\widehat H\oplus\Sigma\longrightarrow\widetilde{\mathcal H}
\]

be `canonical_physical_frame_representative`, with \(q s=1\). The raw
local-Lorentz injection is

\[
L:\Lambda^2V\longrightarrow\widetilde{\mathcal H},
\qquad qL=0.
\]

At the first spinor jet, its independently audited domain is

\[
L_1:S\otimes\Lambda^2V\longrightarrow J^1\widetilde{\mathcal H},
\qquad \dim(S\otimes\Lambda^2V)=32\cdot55=1,760.
\]

Let \(\widetilde F\) be the raw frame-to-output construction before the
canonical input section is imposed, and let \(P_{\mathrm{phys}}\) select only
the finally accepted physical curvature and equation sectors. The exhaustive
section-difference map is

\[
\Delta_L(\lambda;h)=
P_{\mathrm{phys}}\left[
\widetilde F(s(h)+L\lambda)-\widetilde F(s(h))
\right].
\]

Linearity reduces this to

\[
\Delta_L=P_{\mathrm{phys}}\widetilde F L.
\]

Intrinsic local-Lorentz descent requires \(\Delta_L=0\) in exact PBW normal
form for every required Lorentz jet basis vector. This is an upstream
section-independence condition. A nonzero \(\widetilde F L\) cannot be erased
by merely saying that it is "modulo \(K\)": \(K\) maps into \(\widehat H\),
whereas \(\widetilde F L\) is already in the output complex. If an alternate
lift also changes \(\widehat H\) by a target-gauge direction, that additional
map must be typed explicitly and its contribution must vanish through the
separate identity \(F K=0\).

## 3. What is executable now

### 3.1 Local-Lorentz audit

`src/eleven_dimensional_direct_local_lorentz.rs` exhausts all 1,760 basis
columns of \(S\otimes\Lambda^2V\). It checks the direct frame, connection,
anholonomy, and \(J^{(1)}\) routes, including the corrected raised/lowered
spinor bilinears in Eq. (28). Its tracked artifact reports:

* direct frame rank 1,760;
* direct connection rank 1,760;
* direct \(C\)-trace, \(T\)-trace, and \(J^{(1)}\) ranks 32;
* all 1,760 Eq. (28) source columns checked;
* zero mismatches against the source-fixed connection construction;
* a deliberately mixed-index mutation detected on all 1,760 columns.

`src/eleven_dimensional_j1_lorentz_residual.rs` constructs the exact residual

\[
R_J:k^{1760}\longrightarrow k^{32}.
\]

It reports rank 32 and
\(\dim\operatorname{Hom}_{\mathrm{Spin}(1,10)}
(S\otimes\Lambda^2V,S)=1\). Thus the current raw \(J^{(1)}\) response is a
real equivariant obstruction, not a sample artifact. The canonical production
currently avoids it by deleting \(\Psi_{[2]}\) at the input boundary. That
certifies a fixed section only. It does not certify section independence.

This rank-32 result does not by itself prove that the direct physical Riemann,
gravitino-curl, and four-form projections have a nonzero local-Lorentz
residual. The next job must compute those physical projections exhaustively.
Auxiliary X/J/T/W residuals must remain visible, but they are not allowed to
invalidate a physical descent gate unless the accepted physical adapter uses
them without an exact cancellation.

### 3.2 Complete-F production and CUDA substrate

The current CUDA implementation in
`cuda/complete_f_sparse_cuda.cu` and
`src/eleven_dimensional_complete_f.rs` provides:

* exact signed 64-bit sparse application with proved accumulation bounds;
* batched sparse, compact-output, and composed-operator kernels;
* resident operator tables and reusable CUDA contexts;
* CPU/GPU exact equality tests;
* column-sharded COL3 output with semantic and file hashes.

The final measured COL3 production on the RTX 4090 contains 321 of 321
columns and 35,611,900 exact terms. Its operator semantic digest is

```text
efd3a848b64f073a2ab0a9f419bb9515322078dbae99a676576e22cd1b6a01d3
```

The remote measurement root is

```text
/home/brandon/adynkra-runs/complete-f-profile-b8d5237-20260830T1033MDT/
t8-serialized-teardown-wip14
```

This locator is measurement provenance, not yet an authoritative local
descent input. A descent job must ingest immutable local copies and publish a
per-file hash inventory before launch.

The present CUDA contexts accelerate the geometry-level operators during
column construction. They do not yet expose the final COL3 operator as a
general device-resident \(F\) that can consume arbitrary polynomial \(K\)
columns.

### 3.3 Physical K and quotient scaffolds

`src/eleven_dimensional_physical_k.rs` already supplies a strict
`PhysicalKSpecification` boundary. It requires a target parameter domain,
formula, derivative order, authority record, incidence routing, complete-F
digest, and an exact \(F K=0\) certificate. It correctly rejects partial or
unbound claims.

It does not construct \(K\), compose \(F K\), prove the supplied generators
complete, or build the missing Cartesian-to-77-block join. The existing
`eleven_dimensional_level18_target_quotient` code performs exact rank,
kernel, containment, and quotient calculations only after routing is
supplied.

The polynomial harness in
`src/eleven_dimensional_k_fag_solver.rs` understands exact Gaussian-rational
coefficients, eleven momentum variables, channel separation, and the four
solver outcomes. Its recorded basis has 12 leading \(D^{16}\) coefficients
and 44 first-momentum \(pD^{14}\) coefficients. It explicitly stops at
momentum degree one and does not include all \(p^2D^{12}\) or lower symbols.
Its rank-49 bounded negative control is not a generic \(K\) result.

## 4. Exact final-operator size audit

The following counts were obtained by a read-only scan of all 321 final COL3
shards. A polynomial row below means the key
`(exterior-D mask, sector tag, target coordinate)`, with momentum retained in
the coefficient over \(R\).

| tag | sector | exact terms | polynomial rows |
|---:|---|---:|---:|
| 0 | `XTwo` | 112,960 | 19,360 |
| 1 | `XFive` | 1,384,320 | 162,624 |
| 5 | `JMinus` | 153,168 | 15,936 |
| 8 | auxiliary `W2021Raw` | 16,069,472 | 1,652,640 |
| 9 | `LinearizedRiemann` | 1,211,340 | 97,845 |
| 10 | `DirectGravitinoCurl` | 16,473,280 | 877,536 |
| 11 | conditional `DirectCandidateFourForm` | 207,360 | 10,560 |
| **total** |  | **35,611,900** | **2,836,501** |

The exact term distribution is:

| momentum degree | exact terms |
|---:|---:|
| 0 | 16,567,040 |
| 1 | 17,487,280 |
| 2 | 1,557,580 |

| exterior-D degree | exact terms |
|---:|---:|
| 0 | 350,716 |
| 1 | 4,067,936 |
| 2 | 16,278,560 |
| 3 | 14,914,688 |

Every polynomial row has one homogeneous momentum degree. The row counts are
1,839,968 at degree zero, 896,928 at degree one, and 99,605 at degree two.

If tags 9, 10, and 11 become the accepted physical curvature projection, it
contains 17,891,980 exact terms and 985,941 polynomial rows. Tag 11 must not be
included in a final certificate until the \(\Psi_{[3]}\) identification and
relative normalization are fixed.

The target-side free complexes already provide these exact shapes:

| sector | gauge | curvature | Bianchi | curvature to Euler | Euler | Noether |
|---|---:|---:|---:|---:|---:|---:|
| graviton | 66 x 11 | 3,025 x 66 | 9,075 x 3,025 | 66 x 3,025 | 66 x 66 | 11 x 66 |
| four-form | 165 x 55 | 330 x 165 | 462 x 330 | 165 x 330 | 165 x 165 | 55 x 165 |
| Rarita-Schwinger | 352 x 32 | 1,760 x 352 | 5,280 x 1,760 | 352 x 1,760 | 352 x 352 | 32 x 352 |

All consecutive target-side compositions currently vanish exactly. These
complexes are ready to receive the physical source map, but they do not fix
the physical source map or \(K\).

## 5. GPU job A: exhaustive local-Lorentz section difference

### 5.1 Input contract

The job consumes:

1. the exact 55-coordinate \(\Psi_{[2]}\) basis;
2. every required jet basis, beginning with all 1,760
   \(D_\alpha\Psi_{[2]ab}\) columns;
3. two explicit frame lifts, the canonical section and the same section plus
   the selected Lorentz basis input;
4. a frozen physical-sector projection manifest;
5. the complete-F source, binary, operator, target-complex, and convention
   digests.

Linearity permits the production kernel to evaluate only
\(P_{\mathrm{phys}}\widetilde F L e_j\). It must not spend time rebuilding a
generic \(h\) contribution and subtracting two large streams.

### 5.2 Device layout

1. Keep all geometry-level sparse operators resident using the existing
   `ExactCudaSparseOperator` contexts.
2. Batch Lorentz basis columns as lanes. Use structure-of-arrays input keys
   `(lane, source-column, real, imaginary)`.
3. Fuse the frame, connection, J/T/W, and direct physical-adapter stages where
   an intermediate is used by only one successor.
4. Compact only nonzero final physical records. Key them by
   `(lane, sector, coordinate, D-mask, p-exponents)`.
5. Sort and reduce duplicate keys on device before any host transfer.
6. Maintain separate counters for accepted physical sectors and auxiliary
   diagnostic sectors.

The rank-32 \(J^{(1)}\) residual is a mandatory canary. A build that returns
zero on the auxiliary J projection is wrong.

### 5.3 Exactness rule

One nonzero residue at one admissible prime is a valid failure. All-zero
modular output is not by itself an exact proof. A passing job must use one of:

* signed 64-bit exact accumulation with a preflight absolute bound below
  `i64::MAX` for every lane; or
* enough admissible primes that their CRT product exceeds twice the proved
  absolute numerator bound, with every input denominator invertible at every
  prime.

The current three-prime convention can be retained only if its CRT product
satisfies that explicit bound.

### 5.4 Under-30-minute gate

Run a 32-column canary and record expansion products, compaction ratio, peak
VRAM, kernel time, and end-to-end time. Extrapolate to every required Lorentz
jet basis. Launch the exhaustive job only if the bound is below 25 minutes,
leaving five minutes for decoding and exact report publication. If the bound
is larger, split by physical sector without changing the mathematical gate.

This is a performance target, not a present measurement. The existing 1,760
direct diagnostic is executable now, but the physical projected difference
adapter has not been built.

## 6. GPU job B: verify a supplied K

### 6.1 Required K payload

A supplied \(K\) is executable only if its manifest contains:

* the exact \(\Xi_{\mathrm{target}}\) representation, dimension, basis, and
  reducibility maps;
* every sparse Cartesian \(K\) column in the 320-coordinate \(\widehat H\)
  basis;
* every PBW \(D\)-mask and eleven-momentum exponent vector;
* exact Gaussian-rational coefficients and denominator bounds;
* derivative and momentum degree support;
* source or derivation authority;
* the exact complete-F digest to which it applies;
* the basis join and routing into the 77 incidence blocks.

The existing `PhysicalKSpecification` is the metadata boundary, but its
`fk_zero_certificate_sha256` must be produced by this job rather than accepted
as an unchecked assertion.

### 6.2 PBW composition

COL3 entries are differential operators, not ordinary scalar matrix entries.
For a complete-F monomial \(D_Ip^\nu\) and a K monomial
\(D_Jp^\mu\), the worker must compute

\[
(D_Ip^\nu)(D_Jp^\mu)
\]

using the exact superderivative normal form. Simple multiplication of exterior
masks is wrong because crossed or repeated derivatives produce momentum terms
through the Clifford anticommutator.

Precompute the PBW expansion for every encountered pair \((I,J)\). Current F
masks have exterior degree at most three, so the table is driven by the masks
actually present in the supplied K rather than the full \(2^{32}\) space.
Apply F derivatives from right to left to each K polynomial using the same
ordering as repeated `left_multiply_d`.

### 6.3 Device-resident F layout

Add a reusable `CompleteFDeviceOperator` that loads validated COL3 shards into
structure-of-arrays buffers:

```text
source_column:u16
sector:u8
target_coordinate:u32
exterior_mask:u32
momentum_exponents:packed 11-vector
coefficient_real, coefficient_imaginary: exact or modular pair
```

Process one prime at a time if three simultaneous coefficient arrays reduce
useful VRAM. Metadata are shared across primes. The physical projection can
load only tags 9, 10, and the finally authorized four-form tag. The full
auxiliary operator remains a separate diagnostic run.

For each K column:

1. tile its terms against only the matching F source columns;
2. expand through the cached PBW product table;
3. emit keys
   `(K-column, sector, target-coordinate, D-mask, p-exponents)`;
4. sort and reduce on device;
5. stop with failure after publishing the first deterministic nonzero key, or
   finish with an exact zero count.

The final report must also compose the accepted curvature outputs through
their Bianchi, Euler, and Noether maps. These checks prevent a row-tag or
adapter error from producing a vacuous \(F K=0\).

### 6.4 Under-30-minute envelope

For a supplied sparse, low-degree K, verification is a contraction and
canonical reduction problem. It does not require syzygy discovery. The
physical F projection currently has 17,891,980 terms before the four-form
authority gate. A manifest preflight can calculate the exact number of
F-term times K-term products and the worst PBW expansion before launch.

The service-level gate is:

* no more than 20 minutes for all modular contractions;
* no more than five minutes for exact-bound or CRT validation;
* no more than five minutes for artifact validation and atomic publication.

If a supplied K exceeds the measured product envelope, shard by K column,
sector, and prime. Do not weaken exactness or sample parameter components to
meet the time target.

## 7. Why complete-K discovery is a separate problem

After fixing an allowed exterior-D symbol sector, a homogeneous momentum
degree-\(e\) unconstrained vector in \(R^{320}\) has

\[
U_e=320\binom{e+10}{10}
\]

scalar unknowns. Using the measured degree-zero, degree-one, and degree-two
row counts, an unblocked Macaulay construction has the following upper
envelope:

| K momentum degree \(e\) | unknowns \(U_e\) | potential equation rows | replicated F nonzeros |
|---:|---:|---:|---:|
| 0 | 320 | 18,280,106 | 35,611,900 |
| 1 | 3,520 | 107,923,926 | 391,730,900 |
| 2 | 21,120 | 477,663,901 | 2,350,385,400 |
| 3 | 91,520 | 1,723,169,591 | 10,185,003,400 |
| 4 | 320,320 | 5,332,919,592 | 35,647,511,900 |

The equation-row envelope is

\[
1{,}839{,}968\binom{e+10}{10}
+896{,}928\binom{e+11}{10}
+99{,}605\binom{e+12}{10}.
\]

The replicated-nonzero count is
\(35{,}611{,}900\binom{e+10}{10}\). These are conservative full-matrix
counts, but they show why representation decomposition is mandatory. Degree
two already exceeds comfortable 24 GB materialization in a conventional CSR
encoding.

The actual physical K problem is broader because K may carry high exterior-D
symbols such as the recorded \(D^{16}\) and \(pD^{14}\) structures. A complete
search therefore requires:

1. an exhaustive equivariant inventory of allowed D-symbol sectors;
2. decomposition into Spin(1,10) multiplicity blocks;
3. modular generic-rank and blockwise Macaulay kernels;
4. multi-prime agreement of leading monomials and graded Betti data;
5. exact lifting and exact \(F K=0\) verification;
6. minimization and reducibility maps;
7. Hilbert-function equality through a proved regularity bound;
8. a theorem extending equality to every higher degree.

GPU work can plausibly keep each generic-rank sample and many individual
Macaulay blocks below 30 minutes. It cannot turn a finite collection of
bounded kernels into a proof that the entire syzygy module has been found.

## 8. Fail-closed acceptance gates

### 8.1 Local-Lorentz certificate

Pass only if all of the following hold:

1. every required Lorentz jet basis vector is present exactly once;
2. the canonical and varied lift share the same projected \(\widehat H\);
3. every output is in canonical PBW order;
4. the accepted physical-sector difference has zero exact terms;
5. Bianchi and Noether images of the accepted outputs are zero;
6. the known auxiliary \(J^{(1)}\) canary has rank 32;
7. the report names all nonzero auxiliary residual sectors;
8. the authorized four-form and normalization digests are nonnull;
9. CPU exact replay agrees on a deterministic basis subset;
10. the source, binary, operator, basis, and output hashes are bound in the
    final report.

Any omitted Lorentz bidegree yields `incomplete`, not `pass`.

### 8.2 Supplied-K FK certificate

Pass only if all of the following hold:

1. `PhysicalKSpecification` validates without synthetic or control flags;
2. K uses exactly the 320-coordinate \(\widehat H\) basis and never the scale
   column;
3. all K parameter components and reducibility stages are present;
4. complete F has no conditional physical identification;
5. all PBW products are normalized with the pinned Clifford convention;
6. every denominator is admissible at every modular prime;
7. the CRT product exceeds twice the proved numerator bound, or exact signed
   accumulation is used;
8. \(F K\) has zero exact terms in every accepted sector;
9. every target Bianchi and Noether continuation remains zero;
10. the Cartesian-to-77-block join is exact and hash-bound;
11. generic and null-momentum ranks are reported separately;
12. the final certificate binds the complete-F and K semantic digests.

A passing supplied-K certificate proves only
\(\operatorname{im}K\subseteq\ker F\). It does not prove equality.

### 8.3 Complete-K certificate

In addition to 8.2, pass only if:

1. the allowed D and momentum degree range is proved exhaustive;
2. the lifted generators equal the characteristic-zero syzygy module in every
   degree through the bound;
3. the exact Gröbner or equivalent module certificate proves equality above
   the bound;
4. minimal generators and first syzygies are published;
5. unsaturated, irrelevant-ideal-saturated, exceptional-locus, and
   \(p^2=0\) base-change modules are reported separately;
6. no direction appearing only on shell is promoted to target gauge.

## 9. Required mutation tests

Every production binary must make each mutation fail a named gate:

1. flip one Eq. (28) raised/lowered spinor contraction;
2. remove the explicit Lorentz connection term;
3. retain a raw \(\Psi_{[2]}\) coordinate in the canonical section;
4. drop one of the 1,760 Lorentz first-jet columns;
5. change one PBW derivative-ordering sign;
6. suppress one Clifford anticommutator momentum term;
7. include source ordinal 320 in K;
8. relabel auxiliary raw W as the physical four-form;
9. change the four-form relative normalization;
10. delete one K parameter component;
11. replace one exact zero with a residue divisible by one pinned prime;
12. alter one COL3 shard after its manifest is written;
13. substitute synthetic 77-block routing;
14. impose \(p^2=0\) during the off-shell K calculation.

## 10. Provenance and operational contract

Each expensive job must publish, in order:

1. an immutable input manifest;
2. source commit and dirty-diff digest;
3. host binary and CUDA object digests;
4. GPU model, driver, CUDA, Rust, and compiler versions;
5. complete-F, K, basis, convention, target-complex, and prime digests;
6. denominator and accumulation or CRT bounds;
7. five-second machine-readable heartbeat;
8. per-shard checkpoints written to unique paths;
9. exact adoption validation;
10. the final report written atomically and last.

No job may overwrite an authoritative F, K, or descent artifact. A rerun uses
a new run root and may adopt prior immutable shards only after byte, semantic,
schema, ordinal, basis, and source-digest validation.

## 11. Implementation order

1. Freeze the authoritative physical four-form identification and relative
   Riemann/gravitino/four-form normalization.
2. Publish complete physical F with no conditional tag and a stable digest.
3. Refactor final COL3 ingestion into `CompleteFDeviceOperator` without
   changing production bytes.
4. Add the exhaustive local-Lorentz physical projection and run the 32-column
   performance canary, then all required Lorentz jets.
5. Add PBW monomial-pair composition and exact GPU sort-reduce.
6. Verify a supplied K, if one is available, under the 30-minute service
   envelope.
7. In parallel, inventory equivariant K symbol sectors and decompose the
   syzygy problem before launching Macaulay blocks.
8. Prove the regularity or degree bound, lift exact generators, and only then
   publish complete K and the physical quotient.

The immediate coding target is steps 3 through 5. Those steps create reusable
composition infrastructure without prematurely claiming either that the
conditional four-form is physical or that a bounded kernel is complete K.
