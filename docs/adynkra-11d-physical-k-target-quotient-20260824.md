# Physical `K` and the 11D target gauge quotient

Date: 2026-08-24

## Purpose

The completed 77-column computation proves that the repository's canonical
representation-level second-momentum response columns are linearly independent
on the declared partial `F_X` functional slice. It does not determine whether
those responses survive the physical target gauge quotient.

The next semantic task is therefore to construct

```text
K: Xi_target -> H_hat_alpha^a
```

and test the 77 responses modulo `Im(K)`. This record fixes the exact boundary
between what is already known, what the primary sources state, and what remains
to be supplied or derived.

## The four maps must remain distinct

| Symbol | Domain | Codomain | Current status |
|---|---|---|---|
| `G_q` | `Lambda_[q]`, `q = 0,...,5` | `Psi_source_alpha` | Six exact source-gauge maps exist |
| `A_j` | `Psi_source_alpha` | declared `H_hat` response targets | All 77 canonical response columns were computed |
| `K` | `Xi_target` | gamma-traceless `H_hat_alpha^a` | Target domain and exact map are not fixed |
| `F` | `H_hat_alpha^a` | curvature, Bianchi, Euler, and Noether data | `X_[2]` and `X_[5]` slice exists; complete `F` is unfinished |

The six `G_q` domains have Dynkin labels and dimensions

```text
(00000):   1
(10000):  11
(01000):  55
(00100): 165
(00010): 330
(00002): 462
```

They form a direct sum of dimension 1,024. They are inequivalent parameter
domains. Their contributions cannot be treated as six scalar coefficients that
cancel each other before the domains have been identified by explicit maps.
In particular, these six source maps do not by themselves define the target
gauge map `K`.

## What the audited primary sources fix

### arXiv:2002.08502

The Added Note in Proof, Eq. (6.3), proposes an unconstrained spinor
prepotential and states

```text
V = D^alpha Psi_alpha.
```

This is a proposed prepotential relation. It does not print a map from a target
gauge parameter into `H_hat`.

Pinned PDF SHA-256:

```text
62587ef23aa92fd30bb7d978cc4e628275a18dd14fcb10fdf2020906638e554c
```

### arXiv:2007.05097

Eqs. (2.1)-(2.3) fix the gamma-traceless target convention

```text
H_hat = P_320 H
```

and the local gamma-trace redundancy

```text
delta H_beta^b = (gamma^b)_beta^alpha Lambda_alpha.
```

The exact projector satisfies

```text
P_320 gamma Lambda = 0.
```

Consequently, the printed gamma-trace redundancy has zero image in the
gamma-traceless 320-dimensional target. It cannot be reused as a nonzero
physical `K` after projection.

Eq. (2.7) says that a scalar-factorized route `H(V)` would involve fifteen
spinor derivatives, but the exact functional is not printed.

Pinned PDF SHA-256:

```text
197604bc6b5c9e0dfb12044d981aae467920f46554ba9371f1eb9b6389d00a73
```

### hep-th/0101037

Eqs. (24)-(29), (39)-(40), and (44) fix the linearized frame and
anholonomy formulas, the conventional compensator quotient, and the
`X_[2]`, `X_[5]`, `J`, and `W` definitions. The paper still describes `H` as a
semi-prepotential subject to differential constraints that were not known in
that construction. It does not print a fundamental `Psi -> H_hat` target-gauge
map.

Pinned PDF SHA-256:

```text
3d40a1b32fa4491dee56b3e99802172d2c5039b2de198b987ce121a1bbb15cc3
```

## What has now been implemented

`src/eleven_dimensional_physical_k.rs` adds a strict, source-bound input gate
for the missing physical map. A candidate `PhysicalKSpecification` must bind:

1. a distinct target parameter symbol and representation `Xi_target`;
2. its exact dimension and basis;
3. the exact formula `K: Xi_target -> H_hat`, derivative order, signs, and
   rational normalizations;
4. one of three explicit authority paths:
   - a printed primary equation;
   - a bound author-confirmation record;
   - an exact derivation as a kernel of complete `F`;
5. the exact 77-block incidence basis digest;
6. the induced routing into all 77 exact incidence blocks;
7. a physical identification, if any, between `Xi_target` and the six
   independent source domains;
8. for promotion into the completed equation complex, the complete `F`
   operator digest and an exact, bound certificate that `F K = 0`.

### The two counts of 77 are not interchangeable

The 77 second-momentum response columns are the members of the current
operator ansatz. The 77 blocks in
`results/eleven_dimensional_level18_embedded` are exact source-target incidence
maps used by the existing quotient scaffold. They are different typed objects.
The equality of their counts does not identify their bases or their images.

Consequently, the validator requires an explicit induced routing from a
candidate physical `K` into the incidence basis. Until that routing is supplied
and source-bound, the incidence backend is only an exact algebraic scaffold. It
does not quotient the completed 77 response columns by itself.

The validator rejects:

- a synthetic or control routing marked physical;
- the projected gamma-trace redundancy used as a nonzero `K`;
- cancellation between inequivalent source domains;
- a source prepotential silently reused as the target gauge parameter;
- an unbound incidence basis;
- a missing or nonzero `F K` certificate.

An authority-bound `K` can define the target quotient before complete `F` is
available. Promotion of that quotient into the final equation complex also
requires the exact `F K = 0` proof. Only a validated specification reaches the
existing exact rank, kernel, image containment, and quotient backend in
`src/eleven_dimensional_level18_target_quotient.rs`.

The executable audit is:

```bash
target/release/adinkra-codespace \
  adynkra-11d-physical-k-audit \
  results/adynkra_11d_physical_k_determination_audit.json
```

The current audit binds:

- 77 exact incidence blocks;
- direct-sum incidence dimension 439,904;
- typed basis SHA-256
  `150711903df210b9b32e95e83620cb0705c278f5792ba5320179c5f1305e11aa`;
- the four distinct map roles;
- all source statements and PDF hashes above;
- a fail-closed result: no physical `K` is currently validated.

The artifact also records that the incidence blocks are not the
second-momentum response columns and that no physical identification between
those bases is currently available.

Once the missing specification exists, its executable gate is:

```bash
target/release/adinkra-codespace \
  adynkra-11d-physical-k-validate \
  path/to/physical-k-specification.json \
  results/eleven_dimensional_level18_embedded \
  results/adynkra_11d_physical_k_validated.json
```

Unknown JSON fields, stale basis digests, incomplete routing, synthetic inputs,
and partial proof claims fail closed.

## The precise unresolved convention packet

The following questions are sufficient to unblock a source-selected `K`:

1. What is the target gauge-parameter superfield `Xi_target` and its exact
   `Spin(1,10)` representation?
2. What is `delta H_hat` before and after the `P_320` gamma-trace projection?
3. Is `Xi_target` independent of `Psi_source_alpha`?
4. What derivative order, gamma structures, relative signs, and rational
   normalizations define `K`?
5. Are any of the six independent `G_q` domains identified with `Xi_target`?
   If so, what are the exact identification maps?
6. Which complete curvature or equation operator `F` supplies the identity
   `F K = 0`?

There are two honest ways forward:

### Route A: source-selected `K`

Obtain the missing convention from a printed formula or a recorded
confirmation, encode it as `PhysicalKSpecification`, and run the exact quotient.

### Route B: derive `K` from complete `F`

Finish the physical `F`, compute its exact target-side kernel over formal
eleven-dimensional momentum, classify that kernel by `Spin(1,10)` content, and
promote the convention-compatible kernel generators to `K`. This route is
longer but does not guess the gauge image.

## Consequence for the remaining program

The current result does not invalidate the 77-column rank certificate. That
certificate remains an exact lower-bound statement on the declared projected
response slice. The missing `K` prevents the stronger physical statement that
the 77 responses are nontrivial after target gauge quotient.

The safe execution order is now:

1. obtain or derive `K`;
2. validate `K` and compute the exact target quotient;
3. build the companion `p^3 D^11` branch;
4. finish complete `F`, including `J`, torsion, connection, `W`, curvature,
   Bianchi, Euler, and Noether maps;
5. compute the full `F A G_q` system over formal momentum modulo `Im(K)`;
6. solve the exact joint kernel;
7. test any survivor against the `44+84|128` equation complex.

Until step 1 has a real mathematical input, further unquotiented column
production would improve engineering coverage but would not close the main
semantic gap.
