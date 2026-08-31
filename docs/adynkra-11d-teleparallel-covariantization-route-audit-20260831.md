# Teleparallel covariantization and pre-gauge four-form audit

**Date:** 2026-08-31
**Status:** exact route audit, no compensating term adopted
**Question:** can the non-equivariant gauge-fixed teleparallel `D G4` target be
repaired by a local-Lorentz section quotient, or should physical `F` be taken
before gauge fixing?

## 1. Two different covariance problems

Two maps must not be conflated.

The local section-difference map is a vertical gauge test:

```text
omega_L = P_phys F_tilde L.
```

The global Lorentz commutator is an intertwiner test:

```text
chi_X = rho_target(X) T - T rho_source(X).
```

A nonzero `chi_X` is not automatically a local-gauge image. It can be reduced
modulo a local image only after an independently typed local cocycle is proved
to have that image. For the linearized physical four-form around the flat,
zero-four-form background, the primary source says that local image is zero.

## 2. Exact source section and compensator cocycle

Let the raw constrained-frame source be

```text
H_tilde = (H_alpha{}^m, scale, Psi_[2]),
```

and let the current quotient coordinates be

```text
H_phys = (H_hat, scale),  H_hat=P_320 H.
```

The implemented maps are

```text
q(H,scale,Psi_[2]) = (P_320 H,scale),
s(H_hat,scale) = (canonical H_hat representative,scale,0),
q s = 1.
```

`canonical_physical_frame_representative` implements `s` at
`src/eleven_dimensional_h_hat_jet.rs:78-131`.

The linearized local-Lorentz injection is

```text
L Lambda = (0,0,Psi_[2]=Lambda_[2]),
q L = 0.
```

The unit coefficient is exact. The repository audit records

```text
delta H_hat = 0,
delta scale = 0,
delta Psi_[2] = Lambda_[2],
delta Psi_[1,3,4,5] = 0
```

at `src/eleven_dimensional_lorentz_holonomy_compensator_audit.rs:520-526`.
Thus the inhomogeneous source-section cocycle at the linearized flat section is
not unknown:

```text
kappa_L(Lambda;h) = g_Lambda s(h)-s(q g_Lambda s(h)) = L Lambda.
```

At the first spinor jet it is

```text
L_1: S tensor Lambda2 V -> J^1 H_tilde,
D_alpha Lambda_[de] |-> D_alpha Psi_[de],
dimension 32*55 = 1,760.
```

The executable row type and all 1,760 canonical basis keys are already defined
in `src/eleven_dimensional_local_lorentz_descent.rs:17-79` and its exact
section-difference formula is recorded at lines `97-152`.

There is no field-dependent linear cocycle in the current repository that
maps an `H_hat` fluctuation into `Psi_[2]`. Adding such a map would be a new
source transformation law, not a consequence of the audited flat-background
inhomogeneous orbit.

## 3. What the papers require in the physical four-form sector

hep-th/0107155 defines the teleparallel frame and four-form before any Lorentz
connection or Lorentz gauge choice:

```text
E_A = E_A{}^M partial_M,
F_ABCD = (1/6) E_[A A_BCD) - (1/4) C_[AB|{}^E A_E|CD).
```

Its Eq. (2.6a) gives the frame transformation, Eq. (2.6b) gives the
inhomogeneous derivative-of-Lambda transformation of the anholonomy, and Eq.
(2.6c) gives a purely tensorial transformation of `F_ABCD`. In particular,
`F` has no additive derivative-of-Lambda gauge shift. The paper explicitly
states that local Lorentz symmetry is hidden through cancellations in the
teleparallel Bianchi identities.

Consequently, about the flat background used by the repository,

```text
F_background = 0
```

implies

```text
P_F F_tilde L = 0,
P_DF D F_tilde L = 0.
```

For a nonzero background, differentiating a homogeneous tensor transformation
can produce a `(D Lambda) F_background` term. That background-dependent term
is absent here and cannot define a fixed quotient for the present linearized
operator.

The pinned paper hashes already used by the repository are:

```text
hep-th/0107155: 71ccd43c2dea3df8fb9708c016595463cca2674bccad1872c955fc2c8647f25e
hep-th/0101037: 3d40a1b32fa4491dee56b3e99802172d2c5039b2de198b987ce121a1bbb15cc3
```

## 4. Smallest image and quotient bases

### 4.1 Local-Lorentz image in physical `F` and `D F`

The required physical vertical image is dimension zero. The correct target is
not

```text
D G4 / im(omega_L)
```

for a fitted nonzero `omega_L`. It is the full tensorial `D G4` target with
`omega_L=0`.

The 1,760-dimensional vertical source decomposes exactly as

```text
S tensor Lambda2 V = (00001) + (10001) + (01001)
                       32        320       1408.
```

The current auxiliary `J^(1)` cocycle has rank 32 and kernel 1,728. Its image
is the unique gamma-pair contraction

```text
R_(alpha;delta,de)=(109/1056)(Gamma^d Gamma^e)_(alpha,delta).
```

This is certified in
`results/adynkra_11d_j1_lorentz_residual.json`, SHA-256
`7443bfe907215f2b2d326bc0056ad03200d15c5788114321bcacbc04adb74a1b`.
That 32-dimensional image is auxiliary. It is not an allowed physical
four-form quotient and must not be used to erase a `D G4` commutator.

If a raw physical projection is independently proved Lorentz equivariant, one
highest-weight seed in each of the three multiplicity-one source summands is
enough to prove its vertical image zero. Until that equivariance is proved,
the fail-closed basis is all 1,760 Cartesian columns.

### 4.2 Three-form target gauge

If physical `A3` is retained before taking curvature, its Abelian gauge map is

```text
A2 -> A3,  Lambda2 |-> p wedge Lambda2.
```

At any nonzero momentum it has rank 45, with the usual 10-dimensional
first-stage reducibility. The potential quotient has dimension

```text
165-45=120.
```

For the canary momentum `p=e_0`, a smallest explicit image basis is
`A_[0ij]`, `1<=i<j<=10`, of size 45. A quotient basis is `A_[ijk]`,
`1<=i<j<k<=10`, of size 120. The corresponding closed-curvature basis is
`F_[0ijk]`, also size 120. The exact generic-momentum Koszul maps are already
implemented by the four-form target complex in
`src/eleven_dimensional_target_equation_complex.rs:616-635`.

After passing to `G4=p wedge A3`, target gauge has zero image exactly. It also
cannot provide a nonzero quotient in `D G4`.

## 5. Current gauge-fixed target is not a Lorentz intertwiner

The exact witness-source commutator has now been run on source coordinate
`131,857`, corresponding to

```text
outer pair (1,8), momentum axis 5, H_hat ordinal 17.
```

All 55 generators were checked. The unadapted target has 1,032 nonzero
commutator entries. The first is

```text
generator M_02
target coordinate 229
residual 9325/5376 + 7 i/1536.
```

Applying charge conjugation to the output spinor still leaves exactly 1,032
residual entries. Therefore a missing output charge adapter is not the repair.
The durable raw report is
`results/adynkra_11d_teleparallel_d21_lorentz_commutator.json`, SHA-256
`0d0a674d90ff4fe5a80e4861df4e2970ee6896e05ecba65feec54a8e3d5e94cd`.
The independent charge-adapted canary is implemented in
`src/eleven_dimensional_corrected_teleparallel_equivariance.rs`.

This nonzero global commutator cannot be declared zero modulo local Lorentz:
the physical local-Lorentz image is required to be zero at this background.
The result localizes an incomplete or convention-misaligned teleparallel
serialization, source action, or target action.

## 6. Can physical `F` be extracted before gauge fixing?

### 6.1 In the teleparallel paper: yes

The paper treats `A_ABC` and `F_ABCD` as physical superfields on the raw frame.
`F` is computed before Lorentz gauge fixing and is already a Lorentz tensor.
This is the clean covariant route. A local-Lorentz quotient is neither needed
nor allowed after taking `F`.

The smallest exact pre-gauge construction is:

```text
independent A3
  -> G4 = d A3                         [330 x 165 target curvature]
  -> D G4 by exact PBW differentiation [32 x 330 target coordinates]
  -> compare with Eq. (3.1g) gamma-curl expression.
```

It must carry the full raw frame term in the nonlinear theory. At the present
flat linearization, the `C_alpha[b]{}^f F_f[cde]` product in Eq. (3.1g)
vanishes because the background four-form is zero.

### 6.2 In the current `H_hat`-only repository source: not authoritatively

The current source has `320 H_hat + 1 scale` and no independent physical
`A3`. Its closest direct construction is

```text
H_hat -> D H_hat -> Eq. (40) Psi_[3] -> p wedge Psi_[3].
```

This is implemented at
`src/eleven_dimensional_complete_f.rs:1129-1233`. It reads only the projected
`H` jet and does not read `Psi_[2]`, so it is already independent of the local
Lorentz section. Its curvature, Bianchi, Euler, and Noether calculations are
exact.

But hep-th/0101037 presents `Psi_[3]` as a holonomy/conventional-compensator
coordinate eliminated by Eq. (40). It does not identify it with the physical
three-form potential. The repository correctly leaves
`psi_three_identified_as_physical_a3=false`. Thus this route produces a closed
section-independent candidate, not an authoritative physical `F`.

The raw `W2021` output also cannot yet replace it. The production audit keeps
that sector auxiliary because its raw four-form stream is not independently
identified and Bianchi-closed as the physical source map.

Therefore physical `F` can be extracted before gauge fixing only by either:

1. adding the paper's independent `A3/F4` source and using the exact target
   curvature complex; or
2. supplying an authoritative theorem that identifies the Eq. (40)
   `Psi_[3]` candidate with physical `A3` and fixes its normalization.

No local-Lorentz compensator proves option 2.

## 7. Smallest executable comparator

### Gate C0: raw vertical canary

Refactor the direct gravitino route so the inner worker accepts a raw
`LinearizedFrameSuperfields` value without first calling
`canonical_physical_frame_representative`. Existing reusable pieces are:

* `visit_constrained_d_delta`,
  `src/eleven_dimensional_constrained_geometry_jet.rs:397-433`;
* `inject_d_lorentz_compensator_into_d_delta`,
  `src/eleven_dimensional_physical_curvature.rs:1753-1764`;
* `apply_eq25_fermionic_frame`,
  `src/eleven_dimensional_physical_curvature.rs:1017-1085`;
* the exact gravitino curvature in
  `src/eleven_dimensional_complete_f.rs:1261-1364`;
* the teleparallel curl-to-`D F4` operator in
  `src/eleven_dimensional_physical_curvature.rs:482-550`.

Start with `D_0 Psi_[01]=1`, all other raw inputs zero. Stream it through

```text
D Psi_[2] -> D Delta -> Eq25 fermionic frame -> curl -> D G4.
```

One nonzero output is a decisive failure of that raw worker as the physical
section-independent `D G4` map. Do not quotient the output. Add the missing
paper-authorized hidden-covariance term or reject the worker.

If the canary is zero, run all 1,760 basis directions and require exact zero.
After the complete raw worker contains every paper-authorized cancellation, a mutation flipping one `D Psi_[2]` coefficient in exactly one route must become nonzero.

### Gate C1: pre-gauge physical `F`

Add a narrow raw-potential adapter taking one independent `A3` polynomial.
Use the existing target curvature map to produce `G4`, then apply one exact
spinor derivative. Required gates are:

```text
p wedge p wedge Lambda2 = 0,
p wedge G4 = 0,
local vertical residual = 0,
all 165 A3 columns agree with the target curvature matrix,
first descendant agrees with Eq. (3.1g) on every accepted physical source.
```

The first witness should use `p=e_0`, `A_[123]=1`, so the only curvature
component is `G_[0123]` up to the pinned exterior-sign convention. Mutating the
four-form ordinal join or one wedge sign must fail.

### Gate C2: global intertwiner

Only after C0 or C1 is green, repeat

```text
rho_DG4(X) T - T rho_source(X)
```

on source `131,857` and all 55 generators. Required result is zero, not zero
modulo an invented image. Then rerun the bounded coefficient solve.

## 8. Decision

The exact inhomogeneous local-Lorentz compensator is known: it is the unit
`Psi_[2]` shift, with 1,760 first-jet directions. The physical four-form image
of that cocycle must be zero at the present background. The current rank-32
`J^(1)` image is an auxiliary obstruction, not a physical quotient basis.

The non-equivariant gauge-fixed teleparallel target is therefore not rescued
by taking a quotient. The shortest sound route is to extract fundamental
`F=dA3` on the raw frame before gauge fixing and use Eq. (3.1g) as its
first-descendant validation. Within the existing `H_hat`-only source, the
Eq. (40) construction is already section independent but remains only a
candidate until its physical identification is supplied.
