# Roadmap from the 77-column certificate to eleven-dimensional irreducibility

**Status date:** 2026-08-30
**Scope:** Linearized eleven-dimensional spinor-prepotential program
**Status:** Planning and acceptance specification. This document is not itself a physical certificate.

**Repository status:** the p3 portion of Phase 0 is complete. The normalized
rank certificates, denominator-admissibility certificate, and 635-file
production inventory are tracked with the implementation that reproduces
their validation. The broader cross-artifact convention ledger remains future
work. The checkout also contains unrelated modifications that are outside this
roadmap change.

## 1. Executive boundary

The ingested three-prime production run proves an exact and important fact:
the repository's canonical 77-column, one-seed, axis-retained,
second-momentum response matrix has full column rank over three pinned Gaussian
finite fields. The result is rank 77 and nullity 0 at every prime. Phase 0
ingested the complete production evidence and proved that the common cleared
denominator 13,440 is invertible at every pinned prime. A nonzero modular 77 by
77 minor therefore proves that the corresponding 77 characteristic-zero
columns are linearly independent over \(\mathbb Q(i)\).

The computation and its p3 Phase 0 publication gate are complete for the
declared \(p^3D^{11}\) diagnostic branch. This does **not** prove physical
irreducibility. The
missing logical steps are not additional brute-force rank computations on the
same matrix. They are:

1. complete the convention-fixed physical curvature and equation operator
   \(F\) from \(\widehat H_\alpha{}^a\);
2. compute the exact polynomial kernel of completed \(F\), then identify and
   certify the physical target gauge map \(K\) inside it;
3. construct the actual target quotient by \(\operatorname{im}K\);
4. bind the 77 abstract representation-level columns to that physical
   quotient with explicit routing and normalization;
5. compute the exact joint coefficient kernel of the resulting physical
   source-gauge response matrix;
6. if the coefficient kernel is zero, close the declared 77-dimensional
   ansatz as a physical no-go; if a kernel survives, identify every survivor
   and only then test whether a survivor induces exactly one
   \(44+84\mid128\) eleven-dimensional supergravity multiplet.

The critical path therefore has two mathematical promotion gates before the
final matrix calculation:

```text
Phase 2 complete-F gate (Section 8.6): H_hat -> curvature/equation complex
Phase 3 K gate (Section 9.7): exact physical target gauge map and quotient
        |
        v
physical routing of 77 source-gauge response columns
        |
        v
decisive quotient matrix and joint kernel
        +-------------------------+
        | zero                    | nonzero
        v                         v
77-ansatz no-go          survivor classification
                                  |
                                  v
                         44+84|128 comparison
```

The rest of this document turns those statements into executable work,
versioned artifacts, exact acceptance tests, and fail-closed stop rules.

## 2. Claim ladder and exact theorem targets

The word "irreducibility" must be attached to a specific theorem. Four
different statements are in play. They must never be collapsed into one.

### 2.1 Theorem A: finite diagnostic independence

Let \(C\cong\mathbb Q(i)^{77}\) be the coefficient space of the canonical
representation-level second-momentum ansatz, and let

\[
M_{\mathrm{p3,diag}}: C\longrightarrow R_{\mathrm{diag}}
\]

be the declared one-seed, axis-retained diagnostic response. The ingested,
denominator-admissible production proves

\[
\ker M_{\mathrm{p3,diag}}=0.
\]

The analogous checked-in p2 theorem is locally certified. The p3 theorem is
also locally certified after the p3 Phase 0 gate. It says
that no nonzero coefficient vector in the declared 77-dimensional ansatz
vanishes on that diagnostic. It does not say
that the diagnostic is the complete physical curvature complex, that the
ansatz exhausts every allowed operator, or that a target-gauge quotient has
been taken.

### 2.2 Theorem B: physical target-quotient decision for the 77-column ansatz

Let \(\Lambda=\bigoplus_{q=0}^5\Lambda_{[q]}\) be the finite-dimensional
\(\mathbb Q(i)\)-space formed by the six inequivalent source gauge domains,
and define its scalar extension
\(\Lambda_R=R\otimes_{\mathbb Q(i)}\Lambda\). Let \(\mathcal H\) be the polynomial jet
module of the gamma-traceless vector-spinor semi-prepotential
\(\widehat H=P_{320}H\), where \(P_{320}\) is the exact rank-320 projector
from the 352-dimensional vector-spinor ambient space,
let \(\mathcal E\) be the complete physical curvature, Bianchi, Euler, and
Noether module, and let

\[
F:\mathcal H\longrightarrow\mathcal E
\]

be the complete convention-fixed physical operator. Let

\[
K:\mathcal X\longrightarrow\mathcal H
\]

be the physical target gauge map with \(FK=0\), and let
\(G=\bigoplus_qG_q:\Lambda_R\to\Psi_{\mathrm{source}}\). For a coefficient
vector \(c=(c_0,\ldots,c_{76})\), define

\[
A_c=\sum_{j=0}^{76}c_jA_j:
\Psi_{\mathrm{source}}\longrightarrow\mathcal H.
\]

Let \(\pi_K:\mathcal H\to\mathcal H/\operatorname{im}K\). The immediate
decisive coefficient-space map is

\[
\Phi_Q:C\longrightarrow
\operatorname{Hom}_R\left(\Lambda_R,
\mathcal H/\operatorname{im}K\right),
\qquad
\Phi_Q(c)=\pi_KA_cG.
\]

The two outcomes have opposite scientific meanings:

* \(\ker\Phi_Q=0\) means no nonzero member of this exact 77-dimensional
  ansatz maps every source gauge variation into physical target gauge. It is
  a finite-ansatz no-go.
* \(\ker\Phi_Q\ne0\) yields quotient-compatible coefficient candidates. A
  candidate is not yet a physical multiplet. It must pass the full complex
  and cohomology gates below.

If complete \(F\) satisfies \(FK=0\), then full rank of the stacked
\(FA_jG_q\) response is a valid one-way exclusion: a quotient-zero response
would lie in \(\operatorname{im}K\) and hence be annihilated by \(F\). To use
\(\ker F\) itself as the target gauge module, however, the stronger equality
\(\ker F=\operatorname{im}K\) must be proved on the declared graded window.

For every survivor \(c_*\), the inclusion

\[
A_{c_*}\operatorname{im}G\subseteq\operatorname{im}K
\]

is exactly the quotient-zero condition \(\Phi_Q(c_*)=0\); by \(R\)-linearity
it is not a separate stronger hypothesis. It should nevertheless be published
with explicit polynomial witnesses \(A_{c_*}G_q=K\xi_q\), rather than inferred
from dimensions. This inclusion makes \(A_{c_*}\) induce a map
\(\Psi/\operatorname{im}G\to\mathcal H/\operatorname{im}K\). The genuinely
stronger question is whether that induced field-space map is injective or a
cohomological isomorphism. That is distinct from injectivity of the
coefficient map \(\Phi_Q\).

### 2.3 Theorem C: cohomological irreducibility

Even a nonzero kernel of the coefficient decision map is not yet a theorem
that the full supermultiplet is irreducible. For a surviving construction,
construct the relevant
momentum-dependent superspace complex and prove that its gauge-quotiented
cohomology has exactly the expected representation content, with no repeated
copy and no extra class:

\[
H_{\mathrm{phys}}\cong 44\oplus84\mid128.
\]

At a null momentum, the bosonic quotient must contain one graviton little
group representation of dimension 44 and one three-form representation of
dimension 84. The fermionic quotient must contain one gravitino
representation of dimension 128. Supersymmetry must map these sectors into
one another and close modulo gauge and equations. No other cohomology may
remain.

This is the operational irreducibility theorem for a surviving construction.
It cannot follow from a zero coefficient kernel in Theorem B, because that
outcome means the declared ansatz produced no nonzero quotient-compatible
construction to test.

### 2.4 Theorem D: covariant off-shell closure

An off-shell formulation would require a stronger theorem: a covariant
superfield complex closing without imposing the physical equations, together
with a finite or otherwise controlled auxiliary sector. Nothing presently
certifies this. Even a successful Theorem C would establish the irreducible
on-shell physical multiplet reached by the construction, not a finite
auxiliary off-shell formulation.

### 2.5 Required public wording

Until all gates close, use these formulations:

* **Currently proven and locally replayable:** the checked-in bounded
  \(p^2D^{13}\) 77-column diagnostic is full rank at one admissible prime,
  hence independent over \(\mathbb Q(i)\).
* **Proven and locally replayable:** the bounded \(p^3D^{11}\) diagnostic is
  full rank at three pinned admissible primes. The normalized certificates,
  denominator audit, and complete production inventory are tracked. The full
  1.19 GB payload is retained in the content-addressed local archive bound by
  the inventory.
* **Currently supported but incomplete:** the direct Riemann and gravitino
  curvature branches are exact; a closed 330-coordinate four-form candidate
  exists, but its physical identification is conditional.
* **Not yet proven:** the physical target-quotient coefficient decision,
  superspace irreducibility, off-shell closure, or an extension of Einstein's
  equation.

Use the following status labels in technical reports:

| label | status |
|---|---|
| T1a | **PASS:** checked-in bounded p2 77-column diagnostic independence |
| T1b | **PASS:** denominator-admissible bounded p3 77-column diagnostic independence at three primes |
| T2 | **BLOCKED:** physical target-quotient coefficient decision |
| T3 | **PARTIAL:** source cohomology construction |
| T4a | **PASS:** independent target on-shell `44+84|128` reference |
| T4b | **BLOCKED:** source-to-target on-shell cohomology isomorphism |
| T5 | **NOT ESTABLISHED:** covariant off-shell closure |

The lettered theorems and status labels map as follows:

| lettered theorem | status rung |
|---|---|
| Theorem A, finite diagnostic independence | T1a for checked-in p2; T1b for ingested denominator-admissible p3 |
| Theorem B, physical target-quotient coefficient decision | T2 |
| Theorem C, cohomological irreducibility of a survivor | T3 plus the source-linked T4b identification |
| independent target `44+84|128` oracle | T4a; no separate lettered theorem because it is already a reference result |
| Theorem D, covariant off-shell closure | T5 |

The logical structure branches:

```text
T1a/T1b bounded diagnostic evidence      T4a target reference
                  |
                  v
T2 physical quotient decision
       | zero kernel          | nonzero kernel
       v                      v
 finite-ansatz no-go     T3 source cohomology
                                |
                                v
                   T4b source-linked 44+84|128
                                |
                   does not imply
                                v
                    T5 covariant off-shell closure
```

In particular, neither T1a nor T1b implies T2, a nonzero T2 kernel does not imply T3,
dimension counts do not imply irreducibility, T4a does not imply T4b, and an
on-shell T4b result would not imply T5.

## 3. Spaces, maps, gradings, and conventions

### 3.1 Source gauge complex

The six independent source gauge parameter domains are

| form degree \(q\) | B5 label | dimension |
|---:|---|---:|
| 0 | `(00000)` | 1 |
| 1 | `(10000)` | 11 |
| 2 | `(01000)` | 55 |
| 3 | `(00100)` | 165 |
| 4 | `(00010)` | 330 |
| 5 | `(00002)` | 462 |
| **total** |  | **1,024** |

For each \(q\), write

\[
G_q:\Lambda_{[q]}\longrightarrow\Psi_{\mathrm{source},\alpha}.
\]

The six domains are inequivalent and independent. They are not six freely
cancellable coefficients of the physical target map \(K\).

### 3.2 Prepotential response ansatz

The ambient scale must not be confused with the number of operator
coefficients. Eleven-dimensional superspace has 32 odd coordinates. An
unconstrained scalar superfield therefore has

\[
\sum_{n=0}^{32}\binom{32}{n}=2^{32}
\]

theta-monomial component slots, split into \(2^{31}\) even and \(2^{31}\)
odd slots. The gamma-traceless vector-spinor fiber of \(\widehat H\) has
dimension 320. A naive unconstrained tensor product would therefore contain
\(320\cdot2^{32}=1{,}374{,}389{,}534{,}720\) fiber-labeled theta slots before
spacetime jets, equations, and gauge quotients. This number is only an ambient
scale marker. The actual graded jet module \(\mathcal H\) is infinite over the
momentum polynomial ring \(R\), and its relevant finite blocks must be derived
from the declared complex.

The 77 below count candidate equivariant **operators**, not field components.
Consequently, proving independence of 77 columns says nothing about whether
they span the enormous ambient operator module. Phase 5 is the required bridge
from the 77-column inventory to any completeness claim.

The canonical representation-level operator basis has 77 columns. Write

\[
A_j:\Psi_{\mathrm{source},\alpha}\longrightarrow\mathcal H,
\qquad j=0,\ldots,76.
\]

The level-18 inventory contains 42 exact source kernels across 16 irreps and
77 exact embedded source-target incidences. The direct-sum typed incidence
basis has dimension 439,904. These are exact representation data, but their
physical channel routing and coefficients must still be fixed.

### 3.3 Target gauge map

The physical target gauge map is a distinct object:

\[
K:\mathcal X_{\mathrm{target}}\longrightarrow\mathcal H.
\]

Its domain, derivative order, gamma structures, relative signs,
normalization, and routing into the 77-block basis are not fixed by the
currently audited literature. The gamma-trace redundancy removed by
\(P_{320}\) cannot be reused as a nonzero \(K\), because it projects to zero
in \(\widehat H\).

The fail-closed source audit is
`results/adynkra_11d_physical_k_determination_audit.json`, SHA-256
`25e50fee58242408475f46d0c15091647776c54c63414d6af99c351950a3e82f`.
Its `typed_incidence_basis_sha256` field is
`150711903df210b9b32e95e83620cb0705c278f5792ba5320179c5f1305e11aa`;
the corresponding 77-block basis description is stored in
`results/adynkra_11d_level18_target_quotient_basis.json`.
No `results/adynkra_11d_physical_k_validated.json` presently exists.

This missing map must not be confused with the ordinary free component-field
gauge map from diffeomorphism, two-form, and local-supersymmetry parameters
into `(h,A_3,psi)`. The latter is now exact and is recorded in section 4.5a,
but it has no certified lift into `H_hat` and no routing into the 77-block
basis.

### 3.4 Physical operator

The completed operator must have a typed codomain, not a bag of selected
functionals:

\[
F:\mathcal H\longrightarrow
\mathcal E_R\oplus\mathcal E_{G_4}\oplus\mathcal E_\rho
\oplus\mathcal B\oplus\mathcal Q\oplus\mathcal N.
\]

Here:

* \(\mathcal E_R\) contains the complete linearized algebraic Riemann branch;
* \(\mathcal E_{G_4}\) contains the physical closed four-form field strength;
* \(\mathcal E_\rho\) contains the gravitino curl;
* \(\mathcal B\) contains all target Bianchi images;
* \(\mathcal Q\) contains Euler or equation-of-motion images;
* \(\mathcal N\) contains the Noether identities.

The operator must be homogeneous in a declared bigrading

\[
(d_D,d_p)=(\text{exterior spinor derivative degree},
\text{momentum degree}),
\]

with an explicitly declared normal-form order for every \(D_\alpha\) and
\(p_a\).

### 3.5 Coefficient rings

Use three rings for three different jobs:

1. **Exact construction:** \(\mathbb Q(i)\), or an integral Gaussian lattice
   after denominator clearing.
2. **Polynomial module computation:**
   \(R=\mathbb Q(i)[p_0,\ldots,p_{10}]\), with a declared monomial order.
3. **High-volume rank and support checks:**
   \(\mathbb F_{p^2}=\mathbb F_p[i]/(i^2+1)\) for primes \(p\equiv3\pmod4\).

Modular full column rank is a valid characteristic-zero lower-bound proof
after denominator admissibility is checked. Modular nullity is not by itself
an exact characteristic-zero kernel basis. Any claimed survivor must be
lifted and verified exactly over \(\mathbb Q(i)\) or the polynomial ring.

### 3.6 Canonical physical row key

The decisive matrix must retain enough semantics to avoid another opaque hash
projection. Its canonical row identity is:

```text
(source gauge channel q,
 parameter coordinate or certified irrep multiplicity coordinate,
 physical sector,
 target coordinate,
 exterior-D mask,
 momentum monomial,
 p2 or p3 branch,
 retained p3 contraction axis when applicable)
```

The row serializer must order these fields lexicographically in a published
order. A hash bucket may accelerate a lower-bound screen, but it cannot replace
this row identity in the decisive physical artifact.

### 3.7 Required bidegree set and exhaustion obligation

Let

\[
\mathcal B_{\mathrm{req}}
\subset\mathbb Z_{\ge0}^2
\]

be the set of ordered-superderivative and momentum bidegrees
\((d_D,d_p)\) that can occur in the complete physical source-gauge condition
for the declared operator module. The two bounded branches already screened
are

\[
\mathcal B_{\mathrm{scan}}=
\{(13,2),(11,3)\},
\]

corresponding to the \(p^2D^{13}\) and \(p^3D^{11}\) responses after source
gauge composition. At present there is no proof that
\(\mathcal B_{\mathrm{scan}}=\mathcal B_{\mathrm{req}}\).

Phase 5 must determine \(\mathcal B_{\mathrm{req}}\) by one of the following
exact methods:

1. a representation-theoretic decomposition showing that every other
   bidegree has zero equivariant Hom space;
2. a normal-form theorem showing that every other response reduces into the
   scanned branches;
3. a Hilbert-series and regularity certificate giving a finite degree bound,
   followed by complete enumeration through that bound;
4. an exact contracting homotopy or vanishing theorem above a stated degree.

This distinction is asymmetric:

* **No-go branch:** if any subset of required rows already gives rank 77, the
  complete stacked matrix also has rank 77. Therefore a full-rank p2 or p3
  subset is enough to exclude the 77-dimensional coefficient ansatz, provided
  that subset is itself a valid row projection of the complete physical map.
* **Survivor branch:** a vector in the p2 plus p3 kernel is only a
  `scanned-bidegree candidate`. It becomes a physical coefficient survivor
  only after it vanishes at every bidegree in
  \(\mathcal B_{\mathrm{req}}\), or after an exhaustion proof shows that no
  additional bidegree exists.

Every kernel report must therefore carry both `bidegrees_scanned` and
`bidegree_exhaustion_certificate_sha256`. The second field may be null only
for a restricted-scope diagnostic or a one-way full-rank no-go.

## 4. Certified starting state

### 4.1 Three-prime 77-column production

The ingested production record contains all 77 ordinals, all three primes, 231
unique column artifacts, and 132 per-prime reports. Every report passed. The
635-file archive contains 1,191,193,472 bytes, and its source and local
path-size-digest inventories have the identical SHA-256
`a05991d82990175d48c763c3ebe76867baa446ea15a2acec75a26c9296f3d845`.
The frozen earlier tranche was rehashed byte-for-byte before final publication.

The counts derive from, and were verified against, the canonical production
manifest: 77 column ordinals times three primes gives 231 artifacts, while 44
canonical source groups times three primes gives 132 group reports. The
tracked inventory records every relative path, byte length, SHA-256, and role.

**Provenance status:** the full payload is archived outside the worktree, its
complete inventory is tracked, and normalized rank certificates are tracked
under unambiguous prime-slot names. The validation is reproducible with
`scripts/adynkra_11d_p3_phase0_finalize.py`.

The exact p3 pipeline entered in commit `5d9839d`; fused three-prime traversal
entered in commit `7ab0069`. Those commits establish code provenance, not the
missing result-bundle provenance.

The p3 diagnostic has 278,784 rows:

\[
6\text{ gauge channels}\times66\text{ momentum pairs}
\times2\text{ sectors}\times11\text{ retained contraction axes}
\times1\text{ seed}\times32\text{ hash buckets}.
\]

It is distinct from the older 25,344-row \(p^2D^{13}\) diagnostic. Both are
bounded projections. The newer p3 result does not retroactively make either
projection a complete physical map.

| prime slot | prime | rank | nullity | matrix SHA-256 |
|---:|---:|---:|---:|---|
| 0 | 1,073,741,783 | 77 | 0 | `23f21cec2040002338a6913e1f799da4ca3b6094e608affcf3b1034dc9f3b965` |
| 1 | 1,073,741,723 | 77 | 0 | `a004dfe1323b31560470fe6ac3e0bd78f6d7f37cec466e840c2fa76d0f9be1d8` |
| 2 | 1,073,741,719 | 77 | 0 | `949757fcadd88bf88cd70c198c95e96691e05eec154c75da04abe6d9a35dc493` |

The three integers are distinct primes congruent to 3 modulo 4, so the stated
Gaussian finite fields are valid. The exact coefficient audit enumerates
2,877,186 rational coefficient components. Their denominator least common
multiple is 13,440, with gcd 1 against every pinned prime. The exact
coefficient stream SHA-256 is
`22ff3a614d80ca576254c78c46a2429f3aaf684a5f9c044f51fa6f4b2bebb2f7`;
the ordered denominator stream SHA-256 is
`20928c0026fadf660e08d1a1b1b15d6cab903985231d6f62b4494bc7d1cb82c4`.
The recomputed modular flat-plan hashes match every one of the 132 production
reports.

The normalized rank-certificate whole-file SHA-256 values are:

| prime slot | normalized certificate SHA-256 |
|---:|---|
| 0 | `8d624cc932a636eb20772b6aa6ebbd04a3219dd95f1f15c90c08de195434953e` |
| 1 | `8377db3d866e4b012d68aa6635383a1711a4629521cba8ca9df44de70f5830e9` |
| 2 | `1999db473369e0ae78e6dd724ebe996f0b45fb0dd3a6f22dbe3eb681c6b5c701` |

The authoritative remote certificates currently retain historical producer
filenames in which `p0`, `p1`, and `p2` mean **prime slots**, not the
\(p^2D^{13}\) and \(p^3D^{11}\) derivative branches:

```text
/home/brandon/adynkra-runs/
  p3-production-three-prime-fused-20260825T0902MDT/
    p3-all77-rank-p0.json
    p3-all77-rank-p1.json
    p3-all77-rank-p2.json
```

The local normalized copies are
`results/adynkra_11d_p3_all77_rank_prime_slot_{0,1,2}.json`. The inventory
preserves each historical source filename so that `p2` cannot be confused
with the older p2 diagnostic branch.

The Phase 0 production inventory is
`results/adynkra_11d_p3_three_prime_production_inventory_v1.json`, SHA-256
`a790dad975d15892af83bafebbb3ba2a30dcd2c5829d1ffee07928e1bd5b443d`.
The denominator certificate is
`results/adynkra_11d_p3_denominator_admissibility_v1.json`, SHA-256
`4888b47230a2e87c6ee47d64ed176bd3ce677d0f1dda0bcbce601a6b6e2da828`.

The checked-in predecessor is a different branch:

| object | checked-in value |
|---|---|
| artifact | `results/adynkra_11d_second_momentum_full_77_rank_p0.json` |
| branch | bounded \(p^2D^{13}\) partial-`F_X` diagnostic |
| file SHA-256 | `d2d59a078bba548df55b89d66ae500666d07a099e47225d6b8d914a8436c9153` |
| prime | 1,073,741,783 |
| rank/nullity | 77/0 |
| matrix SHA-256 | `87bcd72496b4cf92989f75d20d8188d2159da0226f5ca6c0e77b1815eb266210` |
| static semantic SHA-256 | `eff2e32b1aa7ccb35d89acfe7887e6d3cf482c25dfea6243f73836637b99ed65` |
| artifact commit | `2b02596` |

The p2 certificate and later remote p3 report are chronologically compatible.
They must remain distinct in claims and provenance.

### 4.2 Exact physical branches already available

The present physical-F construction certifies:

* a direct frame-to-Riemann branch with all 1,210 algebraic Riemann
  coordinates;
* a direct gravitino-curl branch with all 1,760 coordinates;
* exact Bianchi checks for both branches;
* computed Euler images and exact Euler-to-Noether checks;
* a conditional hep-th/0101037 Eq. (40) \(\Psi_{[3]}\) candidate mapped into
  all 330 closed four-form coordinates;
* an exact ordered-superderivative normal form;
* a gamma-traceless \(P_{320}\) input section and a canonical local Lorentz
  gauge section;
* separate `X_[2]`, `X_[5]`, `J`, torsion, connection, and raw `W` streams.

The current fail-closed report is
`results/adynkra_11d_complete_physical_f_construction.json`, schema v8,
SHA-256
`1a555acdcdebcade6f51f6c6f60861b63e5998a2fdc093603aed3232dc2820b8`.
Its `passed=true` field means the stated boundary and internal regressions
passed. It explicitly leaves `complete_physical_f_implemented=false` and
`complete_f_operator_sha256=null`.

The compact COL3 production audit is
`results/adynkra_11d_col3_production_audit_20260824.json`, SHA-256
`5d5f7fbd0772521e5db5ffa5d2425184728b5d9f2dd7663864c3cdf5c6071f8f`.
It reports 321 columns, 35,611,900 exact terms, and operator SHA-256
`efd3a848b64f073a2ab0a9f419bb9515322078dbae99a676576e22cd1b6a01d3`.
Its rank-321 result is diagnostic and pre-quotient. The referenced COL3
column-shard directory is not present locally, so from-shards replay also
belongs in Phase 0.

The corresponding tracked compact operator certificate is
`results/adynkra_11d_gauge_fixed_invariant_supercurvature_operator_v4_col3_20260824.json`,
SHA-256
`f215e25920f5c0d21f7cc7a8d996efd89857fb397025fc523af4e81fda853c5a`.

### 4.3 Exact negative result for raw W

Raw \(W\) cannot be silently promoted to the physical four-form. On the
nonzero-\(H\) canary, canonical source-basis ordinal 0 named
`h_spatial_v1_spinor0`:

| quantity | nonzero terms |
|---|---:|
| raw W | 45,260 |
| conditional closed candidate | 648 |
| raw minus candidate | 45,260 |
| raw-W Bianchi residual | 312,704 |

Thus raw \(W\) is neither equal nor proportional to the closed candidate in
the tested convention. The two streams must remain separately tagged until a
source-derived adapter and normalization are proved.

These numbers count nonzero sparse emitted terms for that one exact canary.
They are not representation dimensions, matrix ranks, or term counts for the
complete 321-column operator.

### 4.3a Corrected one-derivative Gamma4 channel diagnostic

The canonical gamma-traceless `H_hat` basis stores a column spinor. The
source-variance-correct Eq. (39) join is therefore

\[
(\Gamma_{[p]}C)C=-\Gamma_{[p]},
\]

not direct application of `Gamma_[p] C` to that basis. With this correction,
the Gamma4 trace and Gamma2 exterior obey the exact identity

\[
\operatorname{Tr}(\Gamma_{[4]}D\widehat H)
=-3\operatorname{Alt}(\Gamma_{[2]}D\widehat H)
\]

on all 10,240 canonical source columns and all 165 three-form rows. Both maps
have rank 165.

The corrected Gamma4 form-vector has measured rank 3,630 and decomposes with
exact ranks `165 + 462 + 3,003`. Reconstruction, hook trace, and hook exterior
residuals are zero. The three raw momentum-to-four-form channels are:

1. `p wedge Gamma4-trace Lambda3`;
2. `i_p Gamma4-exterior Lambda5`, using raised momentum and the mostly-plus
   time sign;
3. `p_e Gamma4-hook H_[4]{}^e`, with no additional metric on the already
   contravariant hook index.

All 2,575,485 exact adapter equivariance checks pass. The three pre-Bianchi
coefficient columns have rank three. The target Bianchi matrix has rank two
and exact kernel `span{(1,0,0)}`: the trace channel has zero Bianchi rows,
while the Lambda5 and hook channels have 2,822,400 and 3,709,440 nonzero rows.

The durable report is
`results/adynkra_11d_raw_three_channel_g4_bianchi.json`, SHA-256
`a0f18ddccaa0c526aa3c38af2ebef081efd1c66d5b6650cfa9f730112c00a6d1`.
Its canonical symbolic-row digest is
`dfe3b010d62c3ec86b65c3d147551ae15edd2986d5f63294da87db1c473b079b`
and its source-module digest is
`c7172139824ef12e3a2557449d849248a4eb88ea6756dd526b22a4b688da881b`.

This closes only the raw one-derivative Bianchi slice. It selects a unique
closed ray up to scale within those three channels. It does not identify that
ray with the physical component four-form, fix its normalization against the
gravitino or graviton, prove exhaustion of every admissible bidegree, impose
the complete source redundancies or target gauge module K, or prove
irreducibility. The earlier teleparallel comparison is bound to the legacy
direct `Gamma_[p] C` convention and must not be used to reject the corrected
trace ray. A fully right-C-corrected descendant comparison was therefore
required.

The corrected descendant comparison is now complete as a bounded no-go test.
The production path first canonicalizes the 320 `H_hat` columns, then composes
the final H-spinor slot of every `DH` and `DDH` slice with the primitive charge
conjugation matrix. It feeds those corrected slices through the unchanged full
Eq. (40) `Psi_[1,3,4,5]` solve, `Delta/DDelta`, Eq. (25) gravitino, curl, and
teleparallel `D G4` chains. Independent parity gates compare the adapted
legacy Gamma2 and Gamma5 maps with direct `-Gamma_[p]` contractions on all
10,240 source columns per degree. Both have zero residual rows, with digests
`5fd9e5af8895e6a4a1b5f020201c6d5f2a7513fc721016234548b295267a2542`
and
`01aa588549f0e3c7dea56c2c539a15e971826636d32974185a8e3f2cc954a3ba`.

Across all 320 columns, the corrected trace-ray candidate has 6,760,512
nonzero rows and the corrected teleparallel descendant has 179,988,160. Their
typed supports have zero rows in common. The candidate scale is therefore
forced to zero, while the target is nonzero, giving 179,988,160 exact residual
rows. The first witness is source column 0, output coordinate 0, ordered
spinor mask `0x00010001`, momentum `p_1`, candidate zero, and teleparallel
coefficient `1/1280`.

The durable report is
`results/adynkra_11d_right_c_full_chain_four_form_normalization.json`, SHA-256
`b171124ca27b79bdf614dedddc8867deab613aa40fb236f66551e8ebf8307ce1`.
Its manifest SHA-256 is
`fddacdeffea8534c23bac7e0b26d8f83b667025cea414f17950cc04b4e9ca072`.
All 320 payload-hashed checkpoints, aggregate counts, stream digests, source
hashes, executable hash, basis hash, target-curvature hash, and teleparallel
map hash were independently re-read and validated.

This result rules out proportionality of the unique closed trace ray to the
pinned teleparallel target on the unrestricted canonical `H_hat` source. It
does not finish physical four-form normalization. Additional source
constraints could change the comparison domain, and the complete equivariant
bidegree inventory could supply channels outside the bounded three-channel
slice. The earlier
`results/adynkra_11d_corrected_lambda3_four_form_normalization_v2.json` is
explicitly marked mixed-convention scratch by its sidecar and is not evidence.

### 4.3b Independent component A3/gravitino target fiber

The physical target normalization now has an independent component-level
anchor that does not call Eq. (25), `H_hat`, Eq. (40), or the gauge-fixed
teleparallel section. It joins two separately typed maps into the canonical
`D_alpha G_[4]` target:

1. the Abelian first-jet map `D A_[3] -> D G_[4]` with `G_4=dA_3`;
2. the component gravitino-curl map fixed by hep-th/0107155v2 Eq. (3.1g),
   with printed coefficient `-1/8` and expanded partition coefficient
   magnitude `1/2`.

For every fixed momentum axis, the independent `D A_3` image has rank 3,840
and the Eq. (3.1g) curl image has rank 1,760. Restricting the latter to the
6,720 target rows outside the `p wedge D A_3` image gives rank 1,440 at all
three pinned primes. The target-image intersection therefore has dimension
320, the combined image has rank 5,280, and the fiber-product kernel has
dimension 1,760. These ranks are identical for all eleven momentum axes.

The 320-dimensional intersection is not inferred only from modular ranks.
For every momentum axis, 320 canonical component-frame curl maps are
independent at all three primes, have zero support outside the `D A_3` image,
and replay exactly over `Q(i)` through an explicit `D A_3` preimage. The
fixed-axis Bianchi residual is zero. A target-form join mutation produces a
nonzero residual. The independent A3 adapter has zero local-Lorentz vertical
image, and no quotient by the unrelated `D Psi_[2]` orbit is applied to the
component curl source.

The durable report is
`results/adynkra_11d_a3_curl_fiber_product.json`, SHA-256
`53f078a1189555734a9c48f674a0528f620460d0ffe8cd60d461f1533b13558a`.
Its source hashes, eleven momentum records, 22 three-prime rank vectors, 11
exact replay records, and normalization canary were independently re-read.
The current fiber-product source hash is
`228a05cdbe2c7b66b7fccbc6ce10eb1ea5146fe7b7cbb75105255070015d3dd8`.

This closes the component A3/gravitino target fiber and its relative
Eq. (3.1g) normalization. It does not identify a map from `H_hat` into this
component fiber, fix normalization relative to the graviton/Riemann branch,
prove complete source or bidegree scope, construct physical K, or prove
irreducibility.

### 4.3c Independent component graviton/gravitino normalization

The remaining target-side relative normalization is now fixed on the ordinary
on-shell component branch, independently of Eq. (25), `H_hat`, and Eq. (40).
The exact comparison starts from hep-th/0101037 Eq. (41),

\[
D_\alpha h_{mn}=i\left[(\Gamma_m C)_{\alpha\gamma}\psi_n{}^\gamma
 +(\Gamma_n C)_{\alpha\gamma}\psi_m{}^\gamma\right],
\]

and applies the independent Pauli-Fierz curvature. After the explicit
derivative-row charge adapter `C^{-1}(C Gamma_m)=Gamma_m`, this agrees exactly
with

\[
D_\alpha R^{\rm repo}_{ab|cd}
=i\left[(p_a\Gamma_b-p_b\Gamma_a)C_{cd}
+(p_c\Gamma_d-p_d\Gamma_c)C_{ab}\right]_\alpha.
\]

The coefficient is one because the repository Riemann convention is twice
the conventional curvature. At each of the eleven coordinate momentum
fibers, the component curl, Riemann descendant, and Eq. (3.1g) `D G_4` maps
have rank 320. The charge adapter, curvature identity, connection-gradient
curl, Riemann Bianchi, and gravitino-curl Bianchi all have zero residual.
Mutation gates detect the overall sign and charge-row join. A separate
fixed-null-momentum implementation checks the same identity on all 352 raw
gravitino-frame columns, also with rank 320, and detects half-normalization,
time-metric, and omitted-pair mutations with 22,496, 1,216, and 11,840
residual entries respectively.

The all-axis durable report is
`results/adynkra_11d_graviton_gravitino_relative.json`, SHA-256
`17f9f227491a1f12bc8449a51f19039522bde86e1a3e1de8250cd9fd01bb11a3`.
The independently derived `p_0` oracle is
`results/adynkra_11d_graviton_relative_oracle.json`, SHA-256
`b03408ee5e3bf7e7156e47636fe189d47a48dc8a20794c7eefec568cdcc8c789`.
Both artifacts were rebuilt report-last after the shared source freeze and
independently re-read with every embedded source hash matching. The oracle
source hash is
`70144d00e4ce3d47301a5ea08490fafb7593b17943075ee8da21253fbbba71dc`.

Together with section 4.3b, this fixes the component target normalization
among Riemann, gravitino curl, and `G_4`. It still does not identify the
corrected Eq. (40) source ray with the independent component `A_3/G_4` fiber,
extend Eq. (41) through off-shell J/X corrections, construct physical K, or
prove irreducibility.

### 4.3d Direct corrected Eq. (40) source-identification no-go

The corrected Eq. (40) `Lambda3` ray has now been compared directly with the
independent component `A_3/G_4` fiber from section 4.3b. This calculation does
not call Eq. (25) or the teleparallel section. It forms

\[
\Psi_{[3]}={1\over16}\Gamma_{[2]}{}^{\beta\gamma}D_\beta
\widehat H_\gamma,
\]

keeps the subsequently applied `D_alpha` as the free target derivative row,
reduces the differentiated expression to PBW normal form, applies
`d:A_3 -> G_4`, and projects each canonical target slice onto the exact
Eq. (3.1g) image. Comparing after `d` annihilates the ordinary
`A_3 -> A_3+p wedge Lambda2` gauge image.

The candidate is Bianchi closed and contains 18,972 nonzero rows over 3,681
PBW monomial slices. Its `(D,p)=(0,2)` branch has 21 slices, all 21 in the
physical image, with zero exact reconstruction residual. Its `(2,1)` branch
has 3,660 slices, none in the physical image, with 869,616 exact residual
rows. The first witness has exterior-spinor mask `3`, momentum axis `10`, and
target coordinate `42`: the candidate coefficient is `1/48`, the reconstructed
physical-image coefficient is `1/336`, and the residual is `-1/56`. The
projector accepts an in-image canary exactly and rejects an off-image mutation.

The durable report is
`results/adynkra_11d_eq40_independent_a3_fiber.json`, SHA-256
`63cdc0edebfe62c1a9d279fa7d1df2d75cc66248a2d8fc513e0fce12147a57ee`.
Its source hash is
`276bacba7c04f04fc2c58517288128d17da7ec54ad4292bc987d279753eec1cc`.
It binds the final physical-fiber report from section 4.3b and the complete
higher-bidegree Hom inventory,
`results/adynkra_11d_higher_bidegree_hom_inventory.json`, SHA-256
`0e595b3787e9d9c1c60090b270bdc7a967efcea064850d9f3531d103b49bb52f`.
All embedded source hashes, branch sums, bidegrees, gauge join, projector
canaries, and the first exact witness were independently re-read.

Therefore the unique corrected Eq. (40) `Lambda3` ray cannot be identified
with the ordinary physical `A_3` on the unrestricted `H_hat` PBW slice. This
is a source-identification no-go, not a proof that physical `F` does not exist.
It does not rule out a constrained source quotient, the gamma-trace spinor ray
in full `H`, higher-bidegree potential maps, or a different physical source
construction.

### 4.3e Engineering-degree source exhaustion

The remaining local polynomial source space at the Eq. (40) engineering
degree is now exhausted. Assigning weight one to `D` and weight two to `p`, a
weight-one `H_hat -> A_3` potential has only the nonnegative bidegree `(1,0)`.
Exact B5 character extraction gives

```text
dim Hom(S tensor H_hat, A_3)       = 1
dim Hom(S tensor S_trace, A_3)     = 1
dim Hom(S tensor full H, A_3)      = 2.
```

The first line is the corrected Eq. (40) ray ruled out in section 4.3d. The
second full-`H` ray factors through `tau=Gamma^a H_a`. The exact rank-32 trace
and rank-320 traceless projectors give `tau(P_320 H)=0`, so this ray vanishes
on every `H_hat` input and cannot change an existing `H_hat` witness row.

At descendant weight four the canonical PBW families are `(4,0)`, `(2,1)`,
and `(0,2)`. The existing exact dimensions for the last two are 52 and 4.
Fresh character extraction gives 49 direct `(4,0)` maps into `D G_4`, split
as `3,10,13,13,10` across `00001,00011,00101,01001,10001`. These maps occupy
a separate PBW row family. They cannot alter the certified `(2,1)` witness
unless a separately proved source differential relation identifies the two
families.

The durable report is
`results/adynkra_11d_eq40_source_exhaustion.json`, SHA-256
`a46e6b42c83354d5a38c8bdeb1d80f35820972de9bc8cc8cf9a7a597503b632a`.
Its source is
`scripts/eleven_dimensional_eq40_source_exhaustion_oracle.py`, SHA-256
`fd54b9643225b1923c6ff8bb61359370c636c81e26e71ae45825334476eb780c`.
It binds the Eq. (40) physical-fiber no-go, the higher-bidegree inventory, the
Clifford projectors, and the three-channel Bianchi kernel, and records semantic
hashes for every character used.

This closes same-engineering local polynomial `H_hat -> A_3` rescue. It does
not close nonlocal inverse-momentum operators, higher-weight potentials,
unknown differential source quotients, or different physical sources. A
direct application of the component Rarita-Schwinger equation to `P_320 H` is
not typed: `H_hat` is a semi-prepotential, while Eq. (25) constructs the
physical component gravitino from `D Delta` and `D Psi`. The smallest honest
on-shell diagnostic must instead use the corrected descendant chain

```text
H_hat -> Eq. (40) Delta -> D Delta -> Eq. (25) psi
      -> gravitino curl -> Rarita-Schwinger Euler.
```

That chain already has the `(2,1)` and `(0,2)` PBW branches, so no additional
outer spinor derivative is allowed. A negative witness is a rank increase
from `C_RS` to `[C_RS;w]`. A positive result requires exact factorization of
the complete residual through both branches of `C_RS`, not a `(2,1)`-only
match. Even a positive result would establish only an on-shell diagnostic,
not an off-shell semi-prepotential constraint.

### 4.4 Exact level-18 representation data

The identity

\[
(11000)\otimes(00001)=
(01001)\oplus(10001)\oplus(11001)\oplus(20001)
\]

has dimensions

\[
429\cdot32=1,408+320+10,240+1,760=13,728.
\]

All 42 required source kernels and all 77 embedded maps have zero exact
raising residual. The current typed basis and quotient backend are therefore
appropriate infrastructure for the eventual physical routing. They do not
determine that routing.

Pinned artifacts:

| artifact | SHA-256 | exact role |
|---|---|---|
| `results/adynkra_11d_level18_embedded_maps.json` | `a5730b5e146c4f5bbabe5db1162a4139e8f2a80b7cb4f6225e2c723f0bb313ac` | 34 abstract pairs, 77 embedded maps, all raising residuals zero |
| `results/adynkra_11d_level18_target_quotient_basis.json` | `0906e965524cd1d4c953eb0e7f610410078b1a66d45b7039d86b08bc31c29253` | exact 77-block, 439,904-dimensional quotient API and synthetic controls |
| `results/adynkra_11d_level18_momentum_validation.json` | `93c04e639ab1f8eaa58dfac35c77fdbc0f3f82bcc77cdd8e9fc7db09a3942cdc` | 42/42 source kernels and 77/77 source-ready copies; full requested step remains false |

The quotient artifact's generic, special, and zero-map tests are synthetic
controls. They prove the API, not the physical target quotient.

### 4.5 On-shell comparison oracle

An independently constructed **target-only reference complex**, which is not
derived from the semi-prepotential or the 77-column ansatz, has null-momentum
physical quotients

```text
graviton:   44
three-form: 84
gravitino: 128
```

and certifies its own light-cone supersymmetry maps and closure residual. This
is only the target comparison oracle. It does not show that the present
covariant semi-prepotential complex maps to, induces, or reproduces those
quotients.

The target oracle is
`results/adynkra_11d_target_equation_complex.json`, SHA-256
`1aa334d1f2cbcc8a46bf2f915b5aeadf16543131241dcdcdc7257e6252d90092`.
It explicitly records that the physical source-to-target F is not
constructed.

The underlying exact free-complex report is
`results/adynkra_11d_free_complex_validation.json`, SHA-256
`d151cfc2b086a737aee7c02f85c6b2b77332451506a8a47cbe78d642876c17e0`.

### 4.5a Physical component gauge complex, distinct from prepotential K

The ordinary free component-field gauge map is now certified as a direct sum
over all eleven formal momentum variables:

\[
K_{\rm comp}:V^*\oplus\Lambda^2V^*\oplus S
\longrightarrow
\operatorname{Sym}^2V^*\oplus\Lambda^3V^*\oplus(V^*\otimes S).
\]

In the repository's unnormalized exterior-product convention, its exact
formulas are

\[
\begin{aligned}
\delta h_{ab}&=p_a\xi_b+p_b\xi_a,\\
\delta A_{abc}&=p_a\Lambda_{bc}-p_b\Lambda_{ac}+p_c\Lambda_{ab},\\
\delta\psi_a{}^\alpha&=p_a\epsilon^\alpha.
\end{aligned}
\]

These are the linearized transformations in arXiv:0903.0259 Eq. (2). The
three-form parameter has the complete second-order reducibility chain

\[
\sigma\mathrel{\mathop{\longrightarrow}^{p\wedge}}\lambda_a
\mathrel{\mathop{\longrightarrow}^{p\wedge}}\Lambda_{ab}
\mathrel{\mathop{\longrightarrow}^{p\wedge}}A_{abc},
\]

with `delta Lambda_ab=p_a lambda_b-p_b lambda_a` and
`delta lambda_a=p_a sigma`. Both consecutive compositions vanish exactly.

The direct component map has shape `583 x 98`. The curvature map into the
ambient ordered-pair Riemann rows, `G_4`, and gravitino curl has shape
`5,115 x 583`. The Riemann block deliberately uses the full `55 x 55 = 3,025`
ordered antisymmetric-pair basis rather than a 1,210-row algebraic-symmetry
projection. Over `Q(i)[p_0,...,p_10]`, every sector has exact
`F_comp K_comp=0`; curvature-Bianchi, curvature-Euler, gauge-Euler, and
Euler-Noether compositions also vanish. Independent one-term gauge-map
mutations produce 100, 8, and 10 residual terms in the graviton, three-form,
and gravitino sectors.

At the fixed lightlike covector `p_a=(1,1,0,...,0)`, the component gauge ranks
are `11,45,32`. The three-form reducibility ranks are `1,10,45`. Quotienting
the raw potential spaces only by component gauge gives dimensions
`55,120,320`. These are not the physical `44,84,128`. The latter are the
on-shell cohomologies

\[
\ker(E_{\rm Euler})/\operatorname{im}K_{\rm comp},
\]

using Euler ranks `11,36,192` and on-shell kernel dimensions `55,129,160`.
Thus the physical bosonic and fermionic totals are both 128, split as
`44+84|128`.

The durable report is
`results/adynkra_11d_physical_component_k.json`, SHA-256
`4b58938dd9861fc27844caf0c68d04617c80abd43d8c7ee4997f59932792c881`.
Its source-module SHA-256 is
`6301d8049528a17935f0b1957c529958a6c4ff0611824264b59401f7d32048ca`.
It binds the target equation complex, free complex, independent `A_3` fiber,
all-axis graviton/gravitino normalization, and independent graviton oracle to
their frozen hashes. An independent reread reproduced every hash, formula,
shape, rank, reducibility identity, cohomology count, formal zero composition,
and mutation requirement.

This closes the free physical component gauge complex only. It does not
construct or infer the prepotential map
`K: X_target -> H_hat`, route component gauge parameters into the 77-block
basis, construct a source superfield equation, add auxiliary fields or
interactions, or establish off-shell closure. Phase 3 therefore remains open.

### 4.6 Certified projectors and missing module promotion

The current Clifford and vector-spinor projector artifact is
`results/adynkra_11d_clifford_projectors_validation.json`, SHA-256
`0f5efdec8c36c7449d27a80be9eab0cd8c39db970b72d9910ae2e6fd385686b2`.
It certifies the 32-dimensional gamma-trace and 320-dimensional
gamma-traceless projectors. The B5-to-Cartesian-Majorana target join is
`results/adynkra_11d_b5_majorana_target_join.json`, SHA-256
`5b7552caa3aad35eafb56fda193b7314b1bcb332a9b754ad0dc9e5c1df92894d`.

These do not supply every product-space projector required by the final
module calculation. In particular, no exported final-composition Cartesian
projectors were found for the relevant `(00002)` and `(11000)` occurrences.

The existing polynomial harness,
`results/adynkra_11d_k_fag_polynomial_harness.json`, SHA-256
`e62c884330f39aabb78da0361d0409695defb3d5b3aaabc64c00b8a54402ec04`,
supports exact polynomial bookkeeping and bounded controls. It does not
contain exported polynomial F blocks, syzygy generators, a quotient
presentation, Hilbert series, Betti table, regularity bound, saturation
certificate, or proved degree bound.

### 4.7 Pure-spinor and T4 boundary

`results/adynkra_11d_covariant_cohomology_gate.json`, SHA-256
`1a5a721864930cc92bad0570e10eb823e26ab2b439238804042208292d2fbc02`,
certifies the degree-two pure-spinor entry gate. It does not certify full
spinorial cohomology or finite-auxiliary off-shell closure.

`results/adynkra_11d_level17_derivative_matrix.json`, SHA-256
`630fc8701ef7d93e9ce37cdd50be4863dba8b062b8c0112a5d88277d8ec4f5cd`,
has a zero-momentum `7x12` matrix of rank 7 and nullity 5. Missing incoming
Here the displayed matrix is the outgoing differential
\(d_{16}:E_{16}\to E_{17}\). The missing `d15` data means the incoming
differential \(d_{15}:E_{15}\to E_{16}\) has not been computed, so this
nullity is not a five-dimensional cohomology theorem.

## 5. Work breakdown and dependency graph

The project is divided into ten phases. Phases 1 through 4 are required before
any physical coefficient decision. Phase 5 is required for a survivor claim
and for any claim that the 77 columns exhaust the complete operator module.
A restricted-scope no-go for the declared 77 coefficients can bypass Phase 5
once a valid physical row subset has rank 77. Phase 6 binds the already
completed p3 data into the promoted physical basis. Phases 7 through 9 contain
the branching decision and scientific interpretation. Phase 10 packages the
result.

```text
0 Provenance freeze
  |
  +------------------------------+
  |                              |
1 Convention and basis contract  2 Complete physical F
  |                              |
  +------------+-----------------+
               |
3 Compute ker F, identify physical K, and build quotient
               |
4 Export exact Cartesian module matrices and projectors
               |
6 Bind and compose the 77 columns in the physical quotient
               |
7 Compute scanned-bidegree coefficient kernel
        | rank 77                         | rank < 77
        v                                 v
restricted 77-ansatz no-go       5 Prove bidegree bounds,
        |                           Hilbert data, and completeness
        |                                 |
        |                         scan every required bidegree
        |                                 |
        |                         8 Verify or eliminate survivors
        |                                 |
        |                         9 Compare cohomology with 44+84|128
        +----------------------+----------+
                               |
10 Publish theorem, certificates, and reproduction bundle
```

## 6. Phase 0: freeze provenance and establish one authoritative ledger

**P3 production gate status:** complete as of 2026-08-30. The broader ledger
items for complete F, physical K, source documents, and future quotient
artifacts remain open and are inputs to Phases 1 through 3. This distinction
prevents the closed p3 evidence gate from being mistaken for a completed
physical-map ledger.

### 6.1 Objective

Create a single machine-readable ledger that binds every input to an exact
byte digest, semantic digest, source commit, command line, compiler, binary,
schema, basis ordering, and mathematical role.

### 6.2 Required inputs

At minimum:

* the three all-77 rank certificates and logs;
* all 231 p3 column artifacts and 132 reports;
* the production run manifest, binary digest, source digests, and CUDA build
  metadata;
* `results/adynkra_11d_complete_physical_f_construction.json`;
* the v4/COL3 invariant-supercurvature report and column shards;
* `results/adynkra_11d_level18_embedded_maps.json`;
* `results/adynkra_11d_level18_target_quotient_basis.json`;
* `results/adynkra_11d_physical_k_determination_audit.json`;
* the exact Majorana, Clifford, free-complex, target-equation, and B5 join
  reports;
* all source PDFs used to fix a coefficient or convention, with SHA-256.

### 6.3 Ledger schema

Each entry must contain:

```json
{
  "logical_id": "stable role name",
  "path": "repository-relative or archived path",
  "sha256": "whole-file hash",
  "semantic_sha256": "canonical decoded mathematical payload hash",
  "schema": "versioned schema string",
  "producer_commit": "full git SHA",
  "producer_diff_sha256": "hash of any dirty diff or null",
  "binary_sha256": "producer executable hash",
  "command": ["exact", "argv"],
  "basis_ids": ["all ordered bases consumed or emitted"],
  "dependencies": ["logical_id entries"],
  "denominator_clearance": {
    "integral_lattice_sha256": "hash after canonical denominator clearing",
    "cleared_denominator_sha256": "hash of ordered cleared denominators",
    "prime_slots": [
      {"prime": 1073741783, "gcd_with_cleared_denominator": 1,
       "admissible": true}
    ]
  },
  "status": "authoritative|superseded|diagnostic|conditional",
  "claim_boundary": "what this artifact does and does not prove"
}
```

### 6.4 Acceptance gates

1. Every authoritative artifact exists locally.
2. Every byte hash and semantic hash reproduces.
3. Every referenced ordinal range is complete and gap-free.
4. No two artifacts claim the same logical role with incompatible basis IDs.
5. Conditional artifacts remain tagged conditional.
6. The ledger is reproducible from a clean checkout.
7. Every modular rank certificate identifies the exact denominator-cleared
   integral matrix or lattice, and records gcd 1 between its cleared
   denominator and every prime used. A prime-field rank is inadmissible until
   this check passes.

### 6.5 Stop rules

Stop immediately on a missing remote artifact, mismatched digest, changed
basis order, unpinned binary, or schema adoption ambiguity. Do not regenerate
over an existing authoritative path. Regeneration must use a new run root.

## 7. Phase 1: convention, basis, and type contract

### 7.1 Objective

Make every later map type-check at the level of representation, basis,
grading, signature, normalization, and quotient section.

### 7.2 Convention record

The convention artifact must pin:

* Lorentz signature `diag(-,+,...,+)`;
* \(\epsilon_{0\ldots10}=+1\);
* charge conjugation matrix \(C\), inverse, transpose symmetry, and all index
  raising rules;
* the 11 gamma matrices and the definition of every antisymmetrized
  \(\Gamma_{[p]}\), including whether brackets contain \(1/p!\);
* the raised-spinor gamma convention;
* momentum Fourier convention and factors of \(i\);
* exterior-spinor derivative ordering and the exact relation
  \(\{D_\alpha,D_\beta\}=i(\Gamma^a)_{\alpha\beta}p_a\) in repository
  normalization;
* local Lorentz gauge section;
* \(P_{320}\) gamma-traceless projector;
* frame versus inverse-frame sign conventions;
* normalized versus unnormalized vector-vector curls;
* every compensator and its gauge transformation.

### 7.3 Basis IDs

Every vector space needs an immutable basis ID derived from a canonical
serialization. Required spaces include:

* 32-component Majorana spinors;
* 11-vector and all \(p\)-form bases;
* 352 vector-spinors and the 320 gamma-traceless subspace;
* 1,210 algebraic Riemann coordinates;
* 330 four-form coordinates;
* 1,760 gravitino-curl coordinates;
* every B5 highest-weight and Cartesian realization used by the 77 blocks;
* every polynomial jet basis by \((d_D,d_p)\);
* every target equation, Bianchi, and Noether basis.

The existing typed incidence basis digest
`150711903df210b9b32e95e83620cb0705c278f5792ba5320179c5f1305e11aa`
must either be adopted exactly or superseded by an explicitly joined basis.

### 7.4 Join matrices

Every abstract B5 basis used in representation computations must have an
explicit exact join to the Cartesian Clifford basis used by \(F\). For a join
\(J_V:V_{\mathrm{B5}}\to V_{\mathrm{Cart}}\), verify:

1. full column rank equal to \(\dim V\);
2. exact intertwining with all Lie algebra generators;
3. exact left inverse on the image;
4. normalization and phase convention;
5. compatibility with the stored highest-weight vector;
6. a semantic digest independent of sparse serialization details.

### 7.5 Deliverables

```text
results/adynkra_11d_irreducibility_conventions_v1.json
results/adynkra_11d_irreducibility_basis_ledger_v1.json
results/adynkra_11d_irreducibility_basis_joins_v1.json
```

## 8. Phase 2: complete the physical operator F

### 8.1 Objective

Construct a single exact, convention-fixed operator from \(\widehat H\) to all
physical curvature and equation sectors, with exact Bianchi and Noether
compositions and no conditional field identification.

### 8.2 Preserve the already certified branches

Do not rewrite the direct Riemann or direct gravitino branches unless a
convention mismatch is found. Adopt them by digest and place them behind the
new typed \(F\) interface. Regression requirements:

* all old exact entries remain byte-identical under projection;
* all Riemann algebraic symmetries and Bianchi identities remain zero;
* the scale canary retains its nonzero Ricci/scalar content;
* the gravitino curl retains its target Bianchi, Euler, and Noether gates;
* local Lorentz gauge-section dependence remains explicitly recorded.

### 8.3 Solve the physical four-form identification ourselves

The missing \(\widehat H\to G_4\) adapter should be derived as an exact
equivariant-map problem rather than guessed from raw \(W\).

#### Step 1: enumerate the ansatz space

For each admissible bidegree \((d_D,d_p)\), enumerate

\[
\operatorname{Hom}_{\mathrm{Spin}(1,10)}
\left(J^{d_D,d_p}(10001),(00010)\right),
\]

where `(10001)` is the gamma-traceless vector-spinor and `(00010)` is the
330-dimensional four-form. Include all independent gamma contractions,
metric contractions, epsilon-dual structures, compensator terms, and
normal-form descendants allowed at that degree. Multiplicities must be
computed, not assumed.

#### Step 2: build exact Cartesian maps

Construct each intertwiner in the same Cartesian Majorana basis as the direct
Riemann and gravitino branches. Verify equivariance against a generating set
of \(\mathfrak{so}(1,10)\), exact rank, and highest-weight normalization.

#### Step 3: impose source and target identities

Solve for coefficients subject to:

1. gamma-trace consistency with \(\widehat H=P_{320}H\);
2. invariance under all declared source redundancies;
3. four-form closure, \(dG_4=0\);
4. compatibility with the exact three-form target reducibility complex;
5. the printed hep-th/0101037 Eq. (40) conventional constraints where
   applicable;
6. compatibility with the direct gravitino descendant under the same
   supersymmetry convention;
7. compatibility with the frame/Riemann branch at the next descendant;
8. Euler-to-Noether zero composition.

These conditions form an exact sparse linear system over \(\mathbb Q(i)\).
Compute its rank and nullity. A one-dimensional solution space fixes the map
up to normalization. Higher nullity requires additional physical constraints,
not an arbitrary basis choice.

#### Step 4: fix normalization

Fix the remaining scale by one of the following, in descending authority:

1. a printed component or superspace equation in the pinned convention;
2. an exact match to the linearized component supersymmetry transformations;
3. an exact match to the standard eleven-dimensional superspace torsion and
   \(G_4\) constraint after a fully explicit basis dictionary;
4. an exact relative closure equation joining the Riemann, gravitino, and
   four-form branches.

Aesthetic normalization or proportionality to raw \(W\) is not acceptable.

#### Step 5: classify raw W

After the physical \(G_4\) map is fixed, decompose raw \(W\) into:

\[
W_{\mathrm{raw}}=
c\,G_4+W_{\mathrm{aux}}+W_{\mathrm{exact}}+W_{\mathrm{obstructed}}.
\]

Compute the projection into closed, exact, and Bianchi-nonclosed components.
The current nonzero Bianchi residual strongly indicates that raw \(W\) is not
the physical field strength by itself.

### 8.4 Complete all target compositions

For every \(F\) sector, explicitly build:

```text
potential -> curvature
curvature -> Bianchi
potential -> Euler/equation
Euler/equation -> Noether
```

Require exact zero for each consecutive composition. The final `F` artifact
must expose both the curvature map and the equation-complex maps. The complete
kernel supplies candidate target-gauge directions. A direction is promoted
to physical \(K\) only after its representation, locality, reducibility, and
gauge interpretation are certified.

### 8.5 Required report fields

The complete-F report must include:

* all domain and codomain basis IDs;
* every sector's exact shape, rank, and semantic digest;
* every source equation and convention used;
* all Bianchi and Noether residual counts, required to be zero;
* the complete operator digest;
* the exact polynomial degree support;
* proof that no conditional tag remains;
* proof that raw \(W\) is not double-emitted as physical \(G_4\);
* proof that the local Lorentz quotient is either intrinsic or represented by
  a declared section plus an exact section-independence certificate.

### 8.6 Complete-F acceptance gate

`complete_physical_f_implemented` may become true only if:

1. Riemann, physical \(G_4\), and gravitino-curl branches are all present;
2. all three have exact target Bianchi certificates;
3. all Euler-to-Noether compositions vanish exactly;
4. every basis join is explicit;
5. all normalizations are source-derived or closure-derived;
6. no conditional physical identification remains;
7. a stable complete-F SHA-256 is published.

## 9. Phase 3: compute the exact polynomial kernel of F and identify K

### 9.1 Why deriving K is preferable

The audited sources do not print a complete target gauge transformation
\(K\). A complete \(F\) lets us compute every local polynomial zero-curvature
direction. Those kernel generators are candidates for target gauge, but
zero-curvature does not by itself imply gauge. They must be classified using
representation type, locality, reducibility, action on potentials, and any
available source authority. If physical \(K\) is defined to be the entire
kernel, that definition and its consequences must be stated explicitly. If a
smaller source-selected \(K\) is used, both \(FK=0\) and the residual quotient
\(\ker F/\operatorname{im}K\) must be reported.

### 9.2 Polynomial matrix construction

Represent \(F\) as a sparse graded matrix over

\[
R=\mathbb Q(i)[p_0,\ldots,p_{10}].
\]

Rows and columns must be homogeneous in momentum degree and carry explicit
Spin(1,10) labels. Clear denominators once, record the content factor, and
work over a primitive Gaussian-integer matrix where possible.

### 9.3 Kernel computation

Compute the graded syzygy module

\[
\operatorname{Syz}(F)=\{x\in R^n:Fx=0\}.
\]

Recommended sequence:

1. exploit Spin(1,10) block decomposition before generic algebra;
2. compute generic-point ranks over several admissible finite fields;
3. obtain candidate degree bounds from Hilbert functions;
4. compute modular syzygies blockwise;
5. repeat at multiple primes and compare leading monomials and graded Betti
   numbers;
6. reconstruct coefficients by Chinese remaindering and rational
   reconstruction;
7. verify every lifted generator exactly over \(\mathbb Q(i)\);
8. minimize the generating set and compute first syzygies among generators;
9. identify reducibility chains, if any.

### 9.4 Saturation and exceptional momentum loci

A pointwise kernel at one momentum is not the physical polynomial kernel.
First compute the local polynomial kernel \(\operatorname{Syz}_R(F)\) over
the unspecialized ring \(R\). Saturation with respect to the irrelevant
momentum ideal may be used to diagnose components supported only at the
origin, but it must be reported separately from the unsaturated kernel.

Do **not** saturate \(\operatorname{Syz}_R(F)\) by the light-cone polynomial
\(p^2\) and then promote the enlarged module into \(\operatorname{im}K\).
Instead, compute the on-shell base change over \(R/(p^2)\) independently and
compare it with the base change of the already certified off-shell gauge
module. Directions appearing only after imposing \(p^2=0\) are on-shell
degeneracies or physical torsion candidates, not target gauge by default.
Distinguish:

* generic off-shell kernel over the fraction field;
* local polynomial kernel over \(R\);
* torsion supported at exceptional momentum loci;
* on-shell kernel after imposing \(p^2=0\).

These modules have different physical meanings. The target gauge map must be
the intended local polynomial module, not an accidental null-momentum rank
drop.

### 9.5 Identify and construct K

Choose a canonical minimal homogeneous generating basis \(\{k_a\}\) for the
classified physical gauge submodule and define

\[
K=[k_1\ \cdots\ k_r].
\]

The `PhysicalKSpecification` must bind:

* the target parameter representation and basis;
* every generator degree;
* all gamma structures and rational coefficients;
* the complete-F digest;
* the exact identity \(FK=0\);
* reducibility maps among gauge parameters;
* routing into the 77-block incidence basis;
* the distinction between source gauge maps \(G_q\) and target gauge \(K\).

### 9.6 Quotient construction

Construct

\[
\mathcal H_{\mathrm{phys}}=\mathcal H/\operatorname{im}K
\]

as a graded presented module, not only as a quotient at sampled momentum.
Publish:

* generators and relations;
* a canonical normal-form reducer;
* Hilbert series and Hilbert polynomial;
* graded Betti table;
* regularity bound;
* torsion and saturation status;
* exact quotient basis or normal-form basis through every degree needed by
  the 77-column composition.

### 9.7 K acceptance gate

The quotient is physical only if:

1. `F K = 0` exactly over the polynomial ring;
2. the domain and reducibility of \(K\) are explicit;
3. the generator set is complete through a proved degree or regularity bound;
4. quotient normal forms are deterministic;
5. basis joins to both Cartesian and 77-block coordinates are exact;
6. generic and null-momentum specializations have documented rank behavior.

## 10. Phase 4: exact Cartesian projectors and module promotion

### 10.1 Objective

Replace every representation-only or specialization-only assertion used by
the final proof with an exported exact matrix in the physical Cartesian basis.

### 10.2 Projector construction

For each required irrep inside a tensor-product ambient space:

1. build the exact Lie algebra generators in the ambient Cartesian basis;
2. use gamma traces, metric traces, antisymmetry, Young symmetries, and Casimir
   polynomials to construct candidate projectors;
3. resolve repeated irreps with highest-weight intertwiners and multiplicity
   space diagonalization;
4. normalize projectors over \(\mathbb Q(i)\);
5. export the sparse matrix and its basis IDs.

The projectors needed by the promotion gate include the occurrences of
`(00002)` and `(11000)` in the **specific ambient module matrices used by the
final composition**. Existing generic Clifford and vector-spinor projectors
do not automatically supply these product-space projectors.

### 10.3 Projector acceptance tests

For every projector \(P_\lambda\):

\[
P_\lambda^2=P_\lambda,
\qquad
[P_\lambda,\rho(X)]=0,
\qquad
\operatorname{rank}P_\lambda=\dim\lambda.
\]

For a complete decomposition:

\[
\sum_\lambda P_\lambda=I,
\qquad
P_\lambda P_\mu=0\quad(\lambda\ne\mu).
\]

Also require exact agreement between the projector image and the stored
highest-weight kernel fixtures.

### 10.4 Matrix export contract

Every promoted module map must export:

* sparse matrix dimensions;
* coefficient domain and denominator content;
* row and column basis digests;
* block labels and multiplicity indices;
* exact rank and nullity when computationally feasible;
* modular rank at at least three admissible primes;
* semantic matrix digest based on sorted nonzero triples;
* source derivation and composition residuals.

### 10.5 Specialization evidence

Any previously observed specialization rank lower bound remains only a lower
bound until the actual matrix, basis, specialization point, and hash are
exported. It cannot substitute for a polynomial module theorem. Promotion
requires the exact matrix plus Hilbert or regularity control showing that the
tested degrees exhaust the relevant module.

An unversioned historical report describes a nontrivial specialization lower
bound for one unresolved large block, but no repository artifact currently
binds its matrix, dimensions, specialization point, or digest. The numerical
values are therefore excluded from this roadmap's evidence base. They may be
restored only by landing the underlying matrix and a content-bound audit in
the Phase 0 ledger. Until then, this report supplies no kernel, Hilbert, or
regularity information.

## 11. Phase 5: prove completeness and degree bounds

### 11.1 Why this phase is necessary

The 77 columns are canonical for the repository's declared inventory, but a
physical irreducibility theorem needs a proof that no allowed operator is
missing at the relevant bidegrees. A full-rank finite matrix proves
independence, not completeness of the ansatz.

### 11.2 Representation-theoretic inventory

For each \((d_D,d_p)\) that can contribute to the physical map:

1. decompose the source jet module into B5 irreps with multiplicity;
2. decompose the target quotient module at the same degree;
3. compute the dimension of every equivariant Hom space;
4. enumerate a basis for each multiplicity space;
5. match every basis vector to a canonical operator column;
6. prove that the union is exactly the 77-column inventory for the target
   theorem, or enlarge the inventory if it is not.

### 11.3 Degree bound

Prove a bound \(d_{\max}\) beyond which no new independent generators can
affect the target cohomology. Acceptable proofs include:

* Castelnuovo-Mumford regularity of the relevant quotient or syzygy module;
* stabilization of a rigorously computed Hilbert function with a certified
  Gröbner basis;
* a representation-theoretic vanishing theorem;
* an exact contracting homotopy above the bound.

Empirical stabilization over a few degrees is not sufficient.

### 11.4 Multi-prime Gröbner protocol

To control large exact computations:

1. choose primes that avoid every cleared denominator and preserve required
   extension-field structure;
2. compute leading monomial ideals at a minimum of three primes;
3. require identical Hilbert series, graded Betti numbers, and leading-term
   patterns;
4. reconstruct a characteristic-zero candidate basis;
5. verify every S-polynomial reduces to zero exactly;
6. prove equality with the full characteristic-zero syzygy module, not merely
   inclusion: compute the exact per-degree kernel Hilbert function of \(F\)
   through the certified regularity bound, compute the Hilbert function of the
   torsion-free submodule generated by the lifted basis, and prove equality in
   every degree through that bound. The exact Gröbner basis and regularity
   theorem must then imply equality in all higher degrees;
7. justify every modular-to-characteristic-zero Hilbert comparison by the
   denominator-admissible reduction theorem being used, and reserve at least
   one unused admissible prime as a holdout check;
8. publish an equality certificate containing both Hilbert functions, the
   regularity bound, the exact initial module, and zero residual for every
   lifted generator and S-polynomial;
9. record unlucky primes and exclude them explicitly.

Agreement of leading ideals, Hilbert series, or Betti tables at three primes
is strong fault detection but is not by itself the characteristic-zero module
equality proof required above.

### 11.5 Completeness acceptance gate

The final composition may be called decisive only after:

* the 77 columns are proven to span the declared physical operator space, or
  the theorem is explicitly restricted to the 77-dimensional subspace;
* the relevant degree range is bounded;
* all repeated-irrep multiplicities are resolved;
* no basis direction exists only in the abstract B5 inventory without a
  Cartesian physical realization.

The phase must publish
`results/adynkra_11d_required_bidegrees_v1.json` containing:

* the complete ordered set \(\mathcal B_{\mathrm{req}}\);
* the source of the finite bound;
* the Hom-space dimension at every bidegree;
* each normal-form reduction or vanishing certificate;
* the Hilbert-series, initial-module, and regularity digests used;
* a final `bidegree_exhaustion_certificate_sha256` consumed by every survivor
  report.

## 12. Phase 6: bind the completed p3 branch to the physical quotient

### 12.1 Objective

Reuse the completed p3 production without recomputing its enormous source
traversal, while replacing its diagnostic consumer with a mathematically
bound physical consumer.

### 12.2 Required adapter

Build an exact adapter for each column \(j\):

\[
\text{p3 source stream}_j
\longrightarrow
\text{physical Cartesian }\mathcal H
\longrightarrow
\mathcal H/\operatorname{im}K
\xrightarrow{\overline F}
\mathcal E.
\]

The adapter must bind:

* source fixture and PBW word identity;
* exact embedded-map digest;
* global ordinal 0 through 76;
* source gauge channel \(q\);
* intermediate irrep and multiplicity copy;
* Cartesian join;
* physical \(K\)-quotient normal form;
* complete-F digest;
* final row basis digest.

### 12.3 Reuse strategy

The existing p3 artifacts may be reused if they preserve enough exact source
semantics to feed a new consumer. If they contain only the old hashed
diagnostic rows, they cannot be inverted. In that case use the durable source
checkpoints or rerun only the contraction stage from canonical reduced source
streams. Repeating the full PBW traversal is the last resort.

### 12.4 Cross-checks

Before accepting the physical composition:

1. reproduce the old diagnostic matrices by projecting the new exact physical
   or source-semantic stream through the old consumer;
2. recover all three old matrix hashes exactly;
3. prove that ordinal and source-stream hashes are unchanged;
4. verify the new quotient response independently at three primes;
5. check at least a small exact rational subset end-to-end.
6. bind every modular artifact to the canonical denominator-cleared integral
   lattice and prove that every pinned prime is coprime to the complete cleared
   denominator. Record the ordered denominator-list hash, integral-lattice
   hash, and per-prime gcd in the physical-composition manifest.

## 13. Phase 7: decisive physical 77-column matrix and joint kernel

### 13.1 Matrix definition

For source gauge channel \(q\) and bidegree \(b\), let \(Q_q^{(b)}\) be the
matrix whose column \(j\) is the canonical quotient normal form of

\[
\pi_KA_jG_q
\in\operatorname{Hom}_R
\left(\Lambda_{[q]},\mathcal H/\operatorname{im}K\right)
\]

at bidegree \(b\). Its rows retain every required parameter component and
every quotient coordinate. The decisive direct-quotient matrix is

\[
M_Q=
\begin{bmatrix}
Q_0^{(b_1)}\\
\vdots\\
Q_5^{(b_1)}\\
Q_0^{(b_2)}\\
\vdots
\end{bmatrix}.
\]

For the full theorem, \(b_1,b_2,\ldots\) range over
\(\mathcal B_{\mathrm{req}}\). For the present bounded diagnostic, they range
only over \(\mathcal B_{\mathrm{scan}}\). The exact coefficient condition is

\[
\ker M_Q
=\bigcap_{b\in\mathcal B}\bigcap_{q=0}^{5}\ker Q_q^{(b)}.
\]

There is a separate complete-curvature response matrix

\[
M_F=\operatorname{stack}_{b,q,j}\left(FA_jG_q\right).
\]

Because \(FK=0\), \(\ker M_Q\subseteq\ker M_F\). Consequently, rank 77 of any
valid row subset of \(M_F\) proves a one-way no-go. Rank deficiency of \(M_F\)
does **not** prove a quotient survivor. A survivor requires zero canonical
normal form in \(M_Q\), equivalently explicit witnesses
\(A_cG_q=K\xi_q\), at every required bidegree. Riemann, \(G_4\), gravitino,
Bianchi, and equation sectors belong to complete \(F\) and validate the
one-way curvature screen; they are not described as being "modulo
\(\operatorname{im}K\)" in their own codomain.

### 13.2 Rank strategy

1. Build the fully quotient-aware p3 branch first, with all physical parameter
   and target multiplicity coordinates retained.
2. Stream the complete sparse matrix independently at each pinned prime.
3. Compute rank with at least two independent elimination implementations.
4. Record pivot rows and a compact nonzero-minor certificate.
5. Compare support masks and per-column semantic hashes across primes.
6. If rank is 77 at one admissible prime, the characteristic-zero coefficient
   kernel is zero for the exact composed response. This is a finite-ansatz
   no-go, not a surviving irreducible construction.
7. Use the other primes as strong implementation and provenance checks.
8. On that full-rank branch, recomputing physical p2 is unnecessary for the
   zero-kernel theorem because stacking more rows cannot create a kernel.
9. If quotient-aware p3 has rank below 77, build the complete physical p2
   branch and compute the kernel of the stacked p2 plus p3 matrix. Label every
   nonzero vector a `scanned-bidegree candidate`, not a survivor.
10. Before survivor promotion, consume the Phase 5 bidegree-exhaustion
    certificate and intersect the candidate kernel with
    \(Q_q^{(b)}\) for every remaining
    \(b\in\mathcal B_{\mathrm{req}}\setminus\mathcal B_{\mathrm{scan}}\).

### 13.3 Kernel strategy

If modular nullity is positive:

1. compute modular kernel bases at all primes;
2. align bases by canonical pivot/free-column form;
3. compare kernel dimension and representation content;
4. reconstruct candidate Gaussian-rational vectors;
5. verify candidates exactly against the characteristic-zero matrix;
6. if reconstruction fails, add primes rather than guessing coefficients.

### 13.4 Required outputs

```text
results/adynkra_11d_physical_77_composition_manifest_v1.json
results/adynkra_11d_physical_77_rank_p0_v1.json
results/adynkra_11d_physical_77_rank_p1_v1.json
results/adynkra_11d_physical_77_rank_p2_v1.json
results/adynkra_11d_physical_77_joint_kernel_v1.json
```

Each report must include the complete-F digest, K digest, quotient digest,
77-inventory digest, all basis IDs, row-family counts, pivot certificate,
matrix semantic digest, and exact claim boundary.

## 14. Phase 8: survivor verification

### 14.1 Zero-kernel path

If rank is 77 and nullity is zero:

1. verify the result with an independently serialized matrix;
2. verify the nonzero minor exactly or reconstruct it sufficiently to prove
   characteristic-zero nonvanishing;
3. mutate one convention-sensitive sign and confirm that a designated canary
   fails;
4. verify that removing any required physical row family is reported as a
   weaker theorem;
5. publish the statement that no nonzero member of the proven 77-dimensional
   ansatz satisfies the complete physical source-gauge condition modulo
   target gauge.

This closes Theorem B on its no-go branch for the declared 77-dimensional
coefficient ansatz without requiring Phase 5: adding more valid physical rows
cannot create a kernel. Any broader statement that the 77 columns exhaust all
allowed operators remains subject to the Phase 5 completeness proof. Phase 9
does not follow on the zero-kernel branch because there is no surviving source
construction.

### 14.2 Positive-kernel path

For every exact survivor \(c\):

1. verify that `bidegree_exhaustion_certificate_sha256` is nonnull and that
   \(c\) vanishes for every \(b\in\mathcal B_{\mathrm{req}}\); otherwise label
   it only a scanned-bidegree candidate;
2. compute its support in the 77-column basis;
3. decompose it by source channel, intermediate irrep, and multiplicity copy;
4. construct the corresponding explicit differential operator;
5. verify all source gauge identities exactly;
6. construct explicit polynomial witnesses \(\xi_q\) satisfying
   \(A_cG_q=K\xi_q\) for every source channel and parameter block;
7. verify its physical quotient normal form vanishes independently of \(F\);
8. verify complete \(FA_cG_q=0\) coefficient by coefficient;
9. test whether it lies in a higher syzygy or unrecorded target gauge image;
10. evaluate it at generic, null, and selected special momenta;
11. determine locality and polynomial degree;
12. classify its Spin(1,10) and little-group representation;
13. decide whether it is physical cohomology, gauge-for-gauge, an auxiliary
    class, exceptional-momentum torsion, or an implementation error.

No survivor may be labeled a new physical field based on modular nullity
alone.

## 15. Phase 9: conditional cohomology and 44+84|128 irreducibility test

This phase runs only for an exact nonzero survivor from Phase 8, or for a
separately supplied source construction that passes the same quotient gates.

### 15.1 Build the on-shell specialization

After the covariant polynomial complex is complete, specialize to a generic
null momentum using a basis change whose determinant is certified nonzero.
Construct the little-group action on the quotient spaces.

### 15.2 Compute physical cohomology

For each relevant degree and parity, compute:

\[
H=\ker(d_{\mathrm{out}})/\operatorname{im}(d_{\mathrm{in}}),
\]

where the maps include source gauge, target gauge, equations, and reducibility
as appropriate. Publish exact dimensions, bases, and representation
characters.

### 15.3 Representation match

Require:

| sector | required physical dimension | multiplicity |
|---|---:|---:|
| graviton | 44 | 1 |
| three-form | 84 | 1 |
| gravitino | 128 | 1 |

The bosonic and fermionic dimensions must both equal 128. Character or highest
weight checks must show the correct little-group irreps, not merely the same
dimensions.

### 15.4 Supersymmetry closure

Construct the induced supercharge maps on cohomology and verify:

\[
\{Q_\alpha,Q_\beta\}
=2(\Gamma^a)_{\alpha\beta}p_a
\text{gauge} + \text{equation terms}.
\]

On cohomology, the residual must vanish exactly. Check that the maps connect
all three physical sectors in one connected representation. A direct sum of
two disjoint copies with total dimension 256 would fail irreducibility even
if each copy separately closed.

### 15.5 Comparison with pure-spinor and superspace results

Use Howe's standard-torsion superspace and Cederwall's pure-spinor cohomology
as on-shell comparison oracles, not as a substitute for the missing
\(\widehat H\) adapter. The relaxed `(11000)` torsion sector is a deformation
or off-shell candidate and must not be counted as an additional physical
state without cohomological proof.

## 16. Phase 10: publication and reproducibility

### 16.1 Reproduction bundle

The final bundle must contain:

* a clean reachable repository commit;
* all exact source files and build instructions;
* complete provenance ledger;
* input PDFs or stable source URLs plus hashes;
* all basis and join artifacts;
* complete F and K specifications;
* quotient, Hilbert, Betti, and regularity artifacts;
* the three physical rank certificates;
* exact kernel or zero-kernel certificate;
* survivor reports;
* final cohomology and supersymmetry closure report;
* a concise theorem statement and a separate limitations statement.

### 16.2 Independent reproduction

Require two levels:

1. **Artifact reproduction:** decode existing artifacts and reproduce every
   hash, rank, and residual without recomputing expensive source traversals.
2. **Scientific reproduction:** from a clean checkout and pinned inputs,
   rebuild representative maps and enough complete sectors to reproduce the
   decisive theorem.

### 16.3 Mutation tests

The final suite must include convention-sensitive negative controls:

* wrong boost sign in gamma lowering;
* wrong \(1/p!\) antisymmetrization;
* changed charge-conjugation raising convention;
* swapped momentum exponent order;
* omitted compensator;
* duplicated raw W as physical \(G_4\);
* one altered 77-column ordinal;
* one altered K generator;
* one removed Bianchi or Noether row family.

Each mutation must fail a named gate before publication.

## 17. Computational architecture

### 17.1 CPU responsibilities

Use exact CPU code for:

* basis and projector construction;
* sparse Gaussian-rational joins;
* Gröbner, syzygy, Hilbert, and Betti calculations after representation block
  decomposition;
* rational reconstruction and exact lifted verification;
* artifact decoding and semantic hashing;
* final theorem checks and mutation tests.

### 17.2 GPU responsibilities

Use the GPU only for uniform high-volume work:

* PBW source traversal and normal-form expansion;
* evaluation of many columns at several primes;
* streamed contraction into fixed row bases;
* batched modular elimination where the matrix shape justifies it;
* repeated generic-point rank sampling.

Do not port symbolic syzygy logic to CUDA before representation block sizes
and actual CPU bottlenecks are measured. The completed p3 run shows that the
production framework is reliable, but the present critical path is
mathematical construction of \(F\), \(K\), joins, and degree bounds.

### 17.3 Job contract

Every expensive worker must have:

* immutable input manifest;
* binary and source digests;
* exclusive output ownership;
* durable checkpoints at a mathematically meaningful boundary;
* atomic publication, with report written last;
* five-second machine-readable heartbeat;
* exact adoption validation;
* no mutation of completed authoritative artifacts;
* bounded memory and disk preflight;
* a canary group that finishes quickly and validates end-to-end semantics.

### 17.4 No blind recomputation

Before any large launch, prove which stage actually needs recomputation:

```text
source traversal -> reduced source stream -> physical adapter -> quotient
-> row encoding -> modular matrix -> rank/kernel
```

If the new work changes only the physical adapter, resume from reduced source
streams. If it changes only quotient normal forms, resume from unquotiented
physical rows. A new complete PBW traversal is justified only if no upstream
semantic artifact exists.

## 18. Proof rules for modular computation

### 18.1 Full-rank rule

After denominator clearing and admissibility checks, a nonzero \(r\times r\)
minor modulo one prime proves that the corresponding characteristic-zero
minor is nonzero. Therefore modular full column rank proves
characteristic-zero independence.

### 18.2 Nullity rule

A positive modular nullity supplies only an upper bound on
characteristic-zero rank and a candidate kernel shape. It may be caused by an
unlucky prime. Require stable nullity at several primes and exact lifting.

### 18.3 Polynomial rule

Point evaluation of a polynomial matrix proves only a specialization rank
lower bound. It cannot prove a polynomial identity or a complete syzygy
module. Exact polynomial verification is mandatory for \(FK=0\), generator
completeness, and degree bounds.

### 18.4 Quotient rule

Never implement a quotient by deleting coordinates unless the deleted
coordinate subspace is proven to equal \(\operatorname{im}K\) in the bound
basis. Use a canonical quotient normal form or an explicit complement with a
certified direct-sum identity.

## 19. Risk register

| risk | consequence | detection | mitigation |
|---|---|---|---|
| Conditional four-form is not physical \(G_4\) | complete F invalid | failure of component or descendant matching | enumerate full equivariant ansatz and solve closure constraints |
| Raw W is reused as physical field strength | nonclosed target and double counting | nonzero Bianchi residual | keep separate tags; project only after physical adapter is proved |
| K derived from incomplete F | quotient too large or too small | K changes when a missing F sector is added | freeze complete-F digest before deriving K |
| `FK=0` promoted to `ker F=im K` | nongauge flat directions discarded | residual `ker F/im K` is nonzero | certify inclusion and equality separately |
| Pointwise kernel mistaken for polynomial gauge | false target quotient | momentum-dependent rank jumps | compute syzygy module and saturation |
| 77 columns are independent but incomplete | overstated theorem | Hom-space dimension exceeds 77 | prove inventory and regularity bound |
| 77 operator columns identified with 77 incidence blocks by count | type-incorrect routing | no exact basis join exists | bind explicit routing matrix and both basis digests |
| Six inequivalent source domains cancel as scalar coefficients | spurious gauge invariance | cross-irrep cancellation appears | test every `G_q` domain independently |
| Abstract B5 basis misjoined to Cartesian physics | wrong signs or routing | generator intertwining residual | exact joins with left inverses and phase records |
| Repeated irreps mixed incorrectly | hidden missing direction | multiplicity-space ambiguity | canonical multiplicity intertwiners and projector tests |
| Modular unlucky prime | false survivor or rank drop | disagreement among primes | at least three primes plus exact lift |
| Hashed diagnostic loses physical semantics | old p3 data cannot be reused | missing reversible row meaning | retain reduced source streams or rerun contraction stage |
| Local Lorentz gauge section contaminates F | section-dependent result | nonzero orbit response | prove descent or include orbit in K |
| Exceptional null momentum creates torsion | spurious physical state | class vanishes generically | saturation and generic/off-shell/on-shell separation |
| Quotient and null-cone specialization assumed to commute | lost or created cohomology | fiber Hilbert function jumps | flatness, saturation, or direct special-fiber calculation |
| Equal dimensions called an irrep match | wrong physical content | character or highest-weight mismatch | exact little-group projectors and intertwiners |
| Light-cone closure called off shell | equations and gauge fixing hidden | closure uses null momentum or EOM | keep T4 and T5 as separate theorems |
| Dirty or unreachable provenance | result cannot be reproduced | clean-checkout audit fails | version all artifacts and publish reachable commit |
| The remote p3 bundle or this untracked roadmap is lost before ingestion | the only present record of the three matrix hashes becomes unverifiable | Phase 0 inventory or remote reachability fails | make two read-only copies immediately, publish a tracked per-file hash inventory, and retain normalized certificate copies |

The three dominant schedule and feasibility risks are:

1. the four-form constraint system has nullity greater than one, leaving the
   physical \(G_4\) adapter and normalization underdetermined;
2. syzygy or Gröbner computation over the 11-variable momentum ring grows
   beyond the planned equivariant block reduction;
3. the completed p3 corpus retained only irreversible diagnostic hash rows,
   forcing contraction-stage or full PBW recomputation for the physical
   consumer.

### Claims that remain forbidden

Until their named gates pass, do not state:

* that eleven-dimensional irreducibility has been proved;
* that complete physical \(F\) or physical \(K\) exists;
* that the 77 columns are full rank after the physical quotient;
* that a finite momentum-order scan settles all orders;
* that the 77-dimensional ansatz is the complete operator module;
* that zero coefficient kernel rules out every 11D prepotential or off-shell
  formulation;
* that a nonzero coefficient kernel is a physical multiplet;
* that the target equation complex is induced from the source
  semi-prepotential;
* that the source construction reproduces `44+84|128`;
* that exact light-cone closure proves covariant off-shell closure;
* that a linearized result is a nonlinear supersymmetric Einstein equation;
* that an invariant action exists without a separate exact variation check.

## 20. Fail-closed decision tree

### 20.1 Four-form gate

* **Unique closed map and authoritative normalization found:** promote it into
  complete F.
* **Unique closed ray found but normalization authority fails:** remain
  blocked at Section 8.6. Publish the ray as conditional, preserve its basis
  and residuals, and seek a component, superspace, or closure normalization.
  Do not promote a convention chosen for convenience.
* **Several maps survive:** compute additional descendant and component
  constraints. Do not choose one by simplicity.
* **No map survives:** report that the present semi-prepotential ansatz does
  not reproduce the physical three-form branch under the declared locality
  and degree assumptions. Revisit the ansatz, compensators, or locality.

### 20.2 K gate

* **Kernel is a clean local polynomial module and every generator has a
  physical gauge interpretation:** construct K and its reducibility.
* **Kernel exists only after base change to \(p^2=0\):** classify it as
  on-shell degeneracy or torsion, not target gauge.
* **Kernel is zero:** the chosen \(\widehat H\) representative has no nontrivial
  target gauge after the fixed quotients. Record this rather than inventing K,
  set \(\mathcal X=0\), \(K=0\), and
  \(\mathcal H/\operatorname{im}K=\mathcal H\). Then continue to the
  projector, routing, and coefficient-decision phases with the zero-K digest.
* **Kernel is mixed:** decompose it into source-authorized local gauge,
  exceptional-locus torsion, and residual zero-curvature directions. Use only
  the certified gauge summand as \(\operatorname{im}K\). Publish
  \(\ker F/\operatorname{im}K\) and keep the residual directions for the
  cohomology analysis.
* **Kernel contains generators that cannot be classified:** remain blocked,
  check missing F sectors and basis joins, and do not set K equal to the whole
  kernel by default.

### 20.3 Final 77 gate

* **Rank 77:** close the proven 77-dimensional ansatz as a physical no-go.
* **Rank below 77 on p2 plus p3:** call the lifted vectors
  scanned-bidegree candidates, complete Phase 5, scan every remaining required
  bidegree, and only then run survivor classification.
* **All scanned modular kernels persist but characteristic-zero lifting
  fails:** no exact survivor exists yet. Increase the CRT modulus under a
  proved coefficient-height bound, audit denominators, and reserve holdout
  primes. If exact lifting still cannot be certified, terminate as blocked,
  not as a survivor or no-go.
* **Rank below 77 on the initial scans but every exact candidate is eliminated
  by later required bidegrees:** publish the final stacked rank-77 certificate
  and close the declared ansatz as a no-go.
* **Rank below 77 after all required bidegrees, but no exact kernel basis or
  quotient witnesses can be certified:** terminate as blocked. Modular
  nullity alone is not a survivor theorem.
* **Prime disagreement:** add primes and audit denominators and serialization.
* **Old diagnostic cannot be reproduced:** stop. The physical adapter is not
  bound to the certified p3 source semantics.

## 21. Immediate execution queue

These are the next concrete tasks in dependency order.

### Task 1: ingest the completed p3 production

**Status:** complete. The authoritative outputs are the three normalized rank
certificates, the denominator-admissibility certificate, and the complete
production inventory listed in Section 4.1. The archive resides at
`~/adynkra-artifacts/p3-production-three-prime-fused-20260825T0902MDT`.

Use this explicit transfer contract:

```text
source host:       brandon@192.168.68.71 (SSH alias: stonkbot)
source root:       /home/brandon/adynkra-runs/p3-production-three-prime-fused-20260825T0902MDT
local archive:     artifacts/adynkra-11d/p3-production-three-prime-fused-20260825T0902MDT
tracked inventory: results/adynkra_11d_p3_three_prime_production_inventory_v1.json
tracked ranks:     results/adynkra_11d_p3_all77_rank_prime_slot_{0,1,2}.json
```

The source root is read-only for this operation. Before transferring payload
files, stream a recursively sorted inventory from the source host to the local
machine. Each inventory entry must include repository-independent relative
path, byte length, whole-file SHA-256, and classified role. The inventory must
list every certificate, report, manifest, log, binary, checkpoint retained as
authority, and column artifact. Counts such as 231 and 132 are acceptance
checks, not substitutes for this per-file inventory.

Copy with an archival, checksum-verifying transport into the new local archive
without `--delete` and without writing under the remote run root. Recompute the
entire inventory locally, compare path, length, and digest entry by entry, then
make a second immutable copy outside the repository worktree. Normalize the
three rank-certificate names by prime slot, preserving their original names in
the inventory. Only after both copies and the comparison pass may the tracked
inventory and normalized rank certificates become Phase 0 authority.

Stop on a missing source root, any unexpected or duplicate relative path, a
changed file during transfer, an inventory mismatch, a certificate whose
embedded matrix hash differs from Section 4.1, or any attempt to regenerate a
missing artifact. Do not modify existing result paths. Until this task passes,
the p3 result remains remote-reported rather than repository-replayable.

This task passed on 2026-08-30. The preceding paragraph remains the stop rule
for any future replacement or regeneration attempt.

As part of the same gate, recover or reconstruct from the canonical source
stream the exact denominator-cleared integral lattice used by the p3 rank
publisher. Hash the ordered cleared denominators and prove gcd 1 with each of
the three pinned primes. Failure to establish this does not invalidate the
remote computation operationally, but it prevents the modular rank from being
promoted to a characteristic-zero proof.

### Task 2: generate the unified convention and basis ledger

Extract conventions and basis hashes from the existing Majorana, Clifford,
physical-curvature, target-complex, level-18, and p3 reports. Fail on any
conflict.

### Task 3: enumerate \(\widehat H\to G_4\) equivariant maps

Build a report listing the Hom-space multiplicity at every admissible
bidegree, the explicit Cartesian intertwiners, their ranks, and exact
equivariance residuals.

### Task 4: solve the four-form constraint system

Impose closure, source invariance, target reducibility, descendant matching,
and Euler/Noether constraints. Report solution rank/nullity and normalization
status.

**Current bounded status:** component target normalization among Riemann,
gravitino curl, and `G_4` is fixed. Closure is solved for the corrected
three-channel one-derivative slice, whose Bianchi kernel is exactly the trace
ray. The direct independent physical-fiber test rules out identifying the
corrected Eq. (40) `H_hat` `Lambda3` ray with ordinary physical `A_3` on the
unrestricted PBW source: all 3,660 `(2,1)` slices are off-image. A constrained
source quotient, the full-`H` gamma-trace spinor ray, higher-bidegree potential
maps, and different source constructions remain open. Source redundancy,
Euler/Noether routing, physical K, and bidegree exhaustion also remain open,
so Task 4 is not complete.

### Task 5: publish complete F v1

Join direct Riemann, physical \(G_4\), and direct gravitino branches into one
typed operator. Run all Bianchi and Noether gates and publish its digest.

### Task 6: export exact polynomial F blocks

Serialize every homogeneous block over \(\mathbb Q(i)[p]\) with basis IDs and
semantic hashes. This is the input to K derivation.

### Task 7: compute and lift the kernel module

Compute modular syzygies, compare primes, reconstruct exact generators,
verify \(FK=0\), determine reducibility, and publish K.

### Task 8: compute quotient Hilbert and regularity data

Construct \(\mathcal H/\operatorname{im}K\), certify normal forms, the
unsaturated local polynomial module, any irrelevant-ideal saturation as a
separately labeled diagnostic, Hilbert series, graded Betti table, and a usable
degree bound. Compute the \(p^2=0\) base change separately and never enlarge K
by light-cone saturation.

### Task 9: finish product-space projectors and basis joins

Export and validate every Cartesian projector and multiplicity intertwiner
needed to route the 77 blocks, including the relevant `(00002)` and `(11000)`
occurrences.

### Task 10: prove the required bidegree set

Execute every item in Sections 11.2 through 11.5: publish
\(\mathcal B_{\mathrm{req}}\), the finite degree or regularity bound, every
source-to-target Hom-space multiplicity, the exact match between multiplicity
bases and operator columns, the characteristic-zero module-equality
certificate, and the bidegree-exhaustion digest. This task may run after a
restricted-scope full-rank no-go, but it is mandatory before any nonzero p2
plus p3 kernel vector is called a survivor or before the 77 columns are called
complete.

This task is the executable Phase 5 work package. Its effort is the explicit
`Phase 5 bidegree exhaustion and ansatz completeness` row in Section 22. It
must also decide whether the original p3 reduced source streams are sufficient
to reproduce the historical diagnostic hashes. If those streams are absent,
schedule and execute a contraction-stage or full PBW recomputation before Task
11. That contingency belongs to the physical-composition effort and may not be
left as an unscheduled stop condition.

### Task 11: bind the physical 77-column composition

Attach every ordinal to the complete-F and K digests, reproduce the old
diagnostic hashes, then publish the three-prime physical matrices.

### Task 12: compute the joint kernel

Run independent streamed eliminations, publish pivot/minor certificates, and
lift every scanned-bidegree candidate exactly. If the p2 plus p3 kernel is
nonzero, intersect it with every remaining required bidegree before survivor
promotion.

### Task 13: branch to no-go closeout or irreducibility closeout

If the kernel is zero, publish a no-go explicitly restricted to the declared
77-dimensional ansatz and the certified physical rows, then stop. Do not call
that result a no-go for the complete operator module unless Task 10 has also
proved the 77-column completeness claim. If a survivor exists, classify it,
compute its on-shell cohomology, verify one
\(44+84\mid128\) multiplet and exact supersymmetry closure, then publish the
theorem with explicit scope.

## 22. Effort bands and critical path

These are planning ranges, not commitments. The uncertainty is mathematical,
not GPU throughput.

| phase | optimistic | expected | principal uncertainty |
|---|---:|---:|---|
| provenance and basis ledger | 1 day | 2 to 3 days | remote artifact transfer and conflicting basis IDs |
| four-form Hom inventory | 2 days | 4 to 7 days | multiplicities and Cartesian joins |
| four-form constraint solve and normalization | 3 days | 1 to 3 weeks | whether current ansatz uniquely reaches physical \(G_4\) |
| complete-F integration | 2 days | 4 to 7 days | section independence and regression volume |
| polynomial K and quotient | 1 week | 2 to 6 weeks | syzygy size, saturation, reducibility |
| projectors and module promotion | 3 days | 1 to 3 weeks | repeated irreps and matrix size |
| Phase 5 bidegree exhaustion and ansatz completeness | 3 days | 1 to 3 weeks | explicit regularity bound, Hom-space enumeration, and unscanned momentum orders |
| physical 77 composition | 4 hours | 1 to 3 days | whether reduced source streams are reusable |
| joint kernel | 2 hours | 1 day | matrix row count and survivor lifting |
| no-go closeout or survivor/cohomology branch | 1 day if zero | 1 to 4 weeks if nonzero | positive kernel or exceptional-momentum classes |

The shortest credible route to Theorem B is therefore several weeks if the
four-form map and K module behave cleanly. Theorem C may take longer. More p3
GPU production is not presently the bottleneck.

An aggressive best case, with a supplied physical four-form convention and
clean equivariant module reduction, is roughly 1.5 to 2 weeks to the T2
decision. A defensible planning band is 3 to 6 weeks. If the four-form adapter
or classification of \(\ker F\) cannot be fixed internally, physical
completion is mathematically unbounded rather than compute-bound.

## 23. Definition of done

### 23.1 The physical target-quotient decision is done when

- [ ] complete F has a stable digest and no conditional sector;
- [ ] physical \(G_4\) identification and normalization are exact;
- [ ] physical K is derived or source-fixed;
- [ ] \(FK=0\) holds exactly over the polynomial ring;
- [ ] quotient generators, relations, Hilbert data, and degree bound are
      published;
- [ ] all 77 columns have explicit Cartesian physical routing;
- [ ] old p3 diagnostic hashes reproduce from the bound source semantics;
- [ ] the denominator-cleared integral lattice and ordered denominator list
      are hashed, and every prime used has certified gcd 1 with the cleared
      denominator;
- [ ] the complete physical matrix has a certified characteristic-zero rank
      and exact coefficient kernel;
- [ ] if the scanned-bidegree kernel is nonzero,
      \(\mathcal B_{\mathrm{req}}\) and its exhaustion certificate are
      published and every additional bidegree has been intersected;
- [ ] independent primes and elimination implementations agree;
- [ ] the characteristic-zero proof is written and machine-checkable.

### 23.2 Irreducibility of a surviving source construction is done when

- [ ] all items above pass and the coefficient kernel is nonzero;
- [ ] the full relevant cohomology is computed, not sampled;
- [ ] the on-shell bosonic content is exactly one `44+84`;
- [ ] the on-shell fermionic content is exactly one `128`;
- [ ] little-group characters or highest weights match;
- [ ] supercharges connect the sectors into one representation;
- [ ] closure holds exactly modulo gauge and equations;
- [ ] every survivor is classified and at least one source construction
      survives all gates;
- [ ] off-shell claims remain explicitly separated unless independently
      proved.

## 24. Primary repository entry points

### Core source modules

```text
src/eleven_dimensional_complete_f.rs
src/eleven_dimensional_physical_k.rs
src/eleven_dimensional_level18_target_quotient.rs
src/eleven_dimensional_level18_embedded.rs
src/eleven_dimensional_second_momentum_fx.rs
src/second_momentum_p3_gpu_production.rs
src/eleven_dimensional_h_hat_jet.rs
src/eleven_dimensional_superderivative_normal_form.rs
src/eleven_dimensional_source_fixed_curvature.rs
src/eleven_dimensional_first_superspace_jet.rs
src/eleven_dimensional_majorana.rs
src/eleven_dimensional_prepotential_gate.rs
src/eleven_dimensional_k_fag_solver.rs
src/eleven_dimensional_level18_momentum.rs
src/eleven_dimensional_covariant_cohomology_gate.rs
```

### Current evidence documents

```text
docs/eleven-dimensional-top-down-gates.md
docs/adynkra-11d-second-momentum-full-rank-20260824.md
docs/adynkra-11d-physical-k-target-quotient-20260824.md
docs/adynkra-11d-pure-spinor-t4-crosscheck-20260825.md
docs/adynkra-11d-direct-riemann-supercurvature-20260824.md
docs/adynkra-11d-clifford-projectors.md
docs/adynkra-11d-level17-hook-derivative-matrix.md
docs/adynkra-11d-level15-bridge.md
docs/adynkra-11d-spinor-prepotential-bridge.md
```

### Current exact artifacts

```text
results/adynkra_11d_complete_physical_f_construction.json
results/adynkra_11d_gauge_fixed_invariant_supercurvature_operator_v4_col3_20260824.json
results/adynkra_11d_col3_production_audit_20260824.json
results/adynkra_11d_level18_embedded_maps.json
results/adynkra_11d_level18_target_quotient_basis.json
results/adynkra_11d_physical_k_determination_audit.json
results/adynkra_11d_clifford_projectors_validation.json
results/adynkra_11d_free_complex_validation.json
results/adynkra_11d_target_equation_complex.json
results/adynkra_11d_b5_majorana_target_join.json
```

### Current regeneration probes

These commands are **writers**, not read-only audits. They exercise existing
gates but do not yet implement the future physical-F, K-module, or physical-77
promotion described above. Never point them at an authoritative tracked path
during Phase 0. The CLIs do not currently expose a check-only mode, so all
outputs below go to a disposable scratch directory. Compare scratch output to
authority only after preserving the authoritative bytes.

```bash
scratch="$(mktemp -d "${TMPDIR:-/tmp}/adynkra-11d-roadmap-audit.XXXXXX")"
trap 'rm -rf "$scratch"' EXIT

# Regenerate the present complete-F boundary report into scratch.
cargo run --release -- adynkra-11d-complete-f-build \
  "$scratch/adynkra_11d_complete_physical_f_construction.json"

# Regenerate the source audit showing that physical K is not yet fixed.
cargo run --release -- adynkra-11d-physical-k-audit \
  "$scratch/adynkra_11d_physical_k_determination_audit.json"

# Validate a future authority-bound K specification against the exact
# embedded basis. This must remain fail-closed until a real spec exists.
cargo run --release -- adynkra-11d-physical-k-validate \
  path/to/physical-k-specification.json \
  results/eleven_dimensional_level18_embedded \
  "$scratch/adynkra_11d_physical_k_validated.json"

# Regenerate existing Clifford and vector-spinor projector checks.
cargo run --release -- adynkra-11d-clifford-verify \
  > "$scratch/adynkra_11d_clifford_projectors_validation.json"

# Regenerate the aggregate bounded top-down status report.
cargo run --release -- adynkra-11d-top-down-build \
  "$scratch/adynkra_11d_top_down.json"

# These hashes are scratch diagnostics only. Phase 0 authority remains the
# previously frozen files plus the content-bound ledger.
find "$scratch" -type f -print0 | sort -z | xargs -0 shasum -a 256
```

P3 Phase 0 replay against the immutable local archive:

```bash
cargo test p3_denominator_admissibility_certificate -- --nocapture

python3 scripts/adynkra_11d_p3_phase0_finalize.py \
  --production-root \
    "$HOME/adynkra-artifacts/p3-production-three-prime-fused-20260825T0902MDT" \
  --preflight-root \
    "$HOME/adynkra-artifacts/p3-g0-20-launch-preflight-20260827T0812MDT" \
  --denominator-certificate \
    results/adynkra_11d_p3_denominator_admissibility_v1.json \
  --normalized-rank-dir results \
  --output "$scratch/adynkra_11d_p3_three_prime_production_inventory_v1.json" \
  --source-raw-inventory-sha256 \
    a05991d82990175d48c763c3ebe76867baa446ea15a2acec75a26c9296f3d845 \
  --local-raw-inventory-sha256 \
    a05991d82990175d48c763c3ebe76867baa446ea15a2acec75a26c9296f3d845

cmp "$scratch/adynkra_11d_p3_three_prime_production_inventory_v1.json" \
  results/adynkra_11d_p3_three_prime_production_inventory_v1.json
```

Focused regression suite:

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

The p3 production CLIs already present are:

```text
adynkra-11d-second-momentum-p3-gpu-plan
adynkra-11d-second-momentum-p3-gpu-status
adynkra-11d-second-momentum-p3-gpu-worker
adynkra-11d-second-momentum-p3-gpu-three-prime-worker
adynkra-11d-second-momentum-p3-gpu-rank
```

Do not launch these again merely to replace the certified rank. The completed
production corpus has been ingested and replayed as specified in Phase 0.
Any replacement run must use a new root and pass the same gate independently.

## 25. Literature roles and boundaries

* [Gates, Hu, and Mak, arXiv:2007.05097](https://arxiv.org/abs/2007.05097)
  fixes important \(\widehat H\), projector, and Weyl-covariance conventions.
  It does not print the complete eleven-dimensional target gauge map K.
* [Gates et al., arXiv:2002.08502](https://arxiv.org/abs/2002.08502) supplies
  the component scan and proposed spinor-prepotential context. Its added note
  gives \(V=D^\alpha\Psi_\alpha\), not a complete physical K or F.
* [Gates and Nishino, hep-th/0101037](https://arxiv.org/abs/hep-th/0101037)
  supplies the linearized frame, anholonomy, conventional constraints, and
  X/J/W definitions used by the current geometry stream. It leaves the
  semi-prepotential's full differential constraints unresolved.
* [Howe, hep-th/9707184](https://arxiv.org/abs/hep-th/9707184) supplies the
  standard on-shell eleven-dimensional superspace constraints and closed
  four-form structure. It is an on-shell comparison oracle, not a printed
  \(\widehat H\) adapter.
* [Cederwall, arXiv:1001.0112](https://arxiv.org/abs/1001.0112) supplies
  pure-spinor cohomological structure useful for checking the final physical
  content. It does not replace the explicit Cartesian joins and target
  quotient needed here.
* [Becker et al., arXiv:2101.11671](https://arxiv.org/abs/2101.11671) and the
  associated \(SL(2,\mathbb C)\times G_2\) formulations provide a component
  and superspace comparison for \(C_3\) and \(G_4\). A complete dictionary
  from the repository's 32-Majorana \(\widehat H\) variables to those
  prepotentials must be built before coefficients can be transferred.

## 26. Final scientific interpretation

The 77-column calculation has already ruled out the simplest hope that the
declared second-momentum operator family contains an unnoticed diagnostic
null combination. Every column contributes an independent direction to the
certified response. That makes a zero final physical kernel plausible and
turns the remaining problem into a sharply defined promotion problem.

The next discovery will come from one of three outcomes:

1. the completed physical F and K quotient preserve rank 77, proving that no
   nonzero member of the entire declared ansatz satisfies the physical
   source-gauge condition;
2. the quotient reveals exact survivors hidden by the old diagnostic, which
   must then be classified as gauge, auxiliary, exceptional-momentum, or
   physical cohomology;
3. the four-form or K derivation fails under the current local ansatz,
   demonstrating that the present semi-prepotential construction is missing
   a field, compensator, nonlocal structure, or different constraint.

All three outcomes are scientifically meaningful. The roadmap is designed so
that none can be mistaken for another.
