# Enlarged-source low-bidegree Hom inventory for the three-form branch

**Date:** 2026-08-31
**Status:** exact representation inventory and executable-oracle design
**Scope:** the ambient vector-spinor `H=(10001)+(00001)` together with
`Psi, Psi_[1], ..., Psi_[5]`, before and after the known source quotients.

## 1. Source and target modules

Write

```text
S       = (00001), dim 32
T       = (10001), dim 320
H       = V tensor S = T + S, dim 352
A3      = Lambda3 V = (00100), dim 165
G4      = Lambda4 V = (00010), dim 330
C       = sum(p=0..5) Lambda^p V, dim 1024
```

Here `Psi=Psi_[0]`. The complete raw coordinate source therefore has

```text
dim(H + C) = 352 + (1+11+55+165+330+462) = 1376.
```

The PBW exterior-spinor factors through degree two are

```text
Lambda^0 S = 1
Lambda^1 S = S
Lambda^2 S = 1 + Lambda3 V + Lambda4 V.
```

The last identity is independently visible in the repository's zero-momentum
Clifford audit: precisely ranks `p=0,3,4` survive an exterior pair of spinor
derivatives.

For two ordinary form factors, the multiplicity of a pure `q`-form is

```text
m_q(r,p) = count(k: q    = r+p-2k)
         + count(k: 11-q = r+p-2k),
0 <= k <= min(r,p).
```

The second term is the epsilon/Hodge channel. Each solution gives one
orthogonal-group intertwiner. This formula gives the coefficient counts below
without choosing gamma-matrix coordinates.

## 2. Exact low-bidegree counts

`d_D` is exterior spinor-derivative degree and `d_p` is formal momentum
degree. Counts are dimensions of Lorentz-equivariant Hom spaces, not source
coordinate dimensions.

| bidegree | source block | Hom to A3 | Hom to G4 |
|---|---|---:|---:|
| `(0,0)` | form compensators `C` | 1 | 1 |
| `(0,1)` | form compensators `C` | 2 | 2 |
| `(1,0)` | full `H=T+S` | 2 | 2 |
| `(1,1)` | full `H=T+S` | 5 | 5 |
| `(2,0)` | form compensators `C` | 9 | 10 |

All omitted source blocks have zero Hom at the displayed bidegree by
spin-statistics parity.

### 2.1 Algebraic and one-momentum form maps

The `(0,0)` maps are the identities

```text
Psi_[3] -> A3
Psi_[4] -> G4.
```

The `(0,1)` potential maps are

```text
p wedge Psi_[2] -> A3
i_(p sharp) Psi_[4] -> A3,
```

and the raw field-strength maps are

```text
p wedge Psi_[3] -> G4
i_(p sharp) Psi_[5] -> G4.
```

Only `p wedge Psi_[3]` is automatically closed as a `G4` map. The direct
`Psi_[4]` and `i_p Psi_[5]` maps require a Bianchi solve and cannot be named a
physical field strength merely because their output type is `(00010)`.

### 2.2 One-spinor maps from the full vector-spinor

Both irreducible summands contribute once:

```text
dim Hom(S tensor T, A3) = dim Hom(S tensor T, G4) = 1
dim Hom(S tensor S, A3) = dim Hom(S tensor S, G4) = 1.
```

Thus the ambient `352` has two A3 rays and two G4 rays. The first A3 ray is
the already checked unique `H_hat` trace/exterior ray. The second is the
ordinary `Gamma_[3] D chi` ray of the gamma-trace spinor `chi`.

At one further momentum, the gamma-traceless block contributes three maps to
each target:

```text
T to A3: Lambda2 wedge, Lambda4 contraction, (10100) hook contraction
T to G4: Lambda3 wedge, Lambda5 contraction, (10010) hook contraction.
```

The trace-spinor block contributes two maps to each target because `S tensor
S` contains forms but no corresponding hook:

```text
S to A3: Lambda2 wedge, Lambda4 contraction
S to G4: Lambda3 wedge, Lambda5 contraction.
```

This proves the `(1,1)` counts `3+2=5`. On the gamma-traceless block, the
existing exact Bianchi matrix has rank two and kernel one among the three raw
G4 maps. The unique closed vector is the `p wedge A3` trace channel.

### 2.3 Two-spinor maps from the form compensators

For source degree `p=0,1,2,3,4,5`, the exact counts are

```text
                    p=0  p=1  p=2  p=3  p=4  p=5   total
D^2 Psi_[p] -> A3     1    1    1    2    2    2      9
D^2 Psi_[p] -> G4     1    1    1    2    3    2     10
```

A minimal invariant basis is indexed by triples `(r,p,k)` with
`r in {0,3,4}` and either `q=r+p-2k` or `11-q=r+p-2k`. The latter triples
mean Hodge dual after the indicated contractions. This is preferable to
hand-naming gamma strings because it makes direct, contracted, and epsilon
channels disjoint by construction.

The A3 triples are:

```text
p0: (3,0,0)
p1: (4,1,1)
p2: (3,2,1)
p3: (0,3,0), (4,3,2)
p4: (3,4,2), (4,4,0; Hodge)
p5: (3,5,0; Hodge), (4,5,3)
```

The G4 triples are:

```text
p0: (4,0,0)
p1: (3,1,0)
p2: (4,2,1)
p3: (3,3,1), (4,3,0; Hodge)
p4: (0,4,0), (3,4,0; Hodge), (4,4,2)
p5: (3,5,2), (4,5,1; Hodge)
```

### 2.4 Higher bidegrees and the spinorial descendant target

For a bosonic `A3` or `G4` target, both Hhat slices `(2,1)` and `(0,2)` have
zero Hom exactly. The central element of Spin acts negatively on Hhat and
positively on an even number of spinor derivatives, any number of vectors,
and either bosonic target.

The normalization comparison does not have a bare `G4` target. Its target is
the spinor-valued descendant `D G4 = S tensor (00010)`. Exact doubled-weight
character convolution and Weyl-alternant extraction instead give

```text
Hom(Lambda2 S tensor V tensor Hhat, D G4), (2,1):
  7(00001) + 7(00011) + 11(00101) + 14(01001) + 13(10001)
  total coefficient dimension 52

Hom(Sym2 V tensor Hhat, D G4), (0,2):
  1(00001) + 0(00011) + 0(00101) + 1(01001) + 2(10001)
  total coefficient dimension 4.
```

For `D A3`, the corresponding totals are 45 and 4. The five target labels in
`D G4` have dimensions `32, 5280, 3520, 1408, 320`, summing to 10,560.
Exact Cartesian Casimir projectors onto all five summands are now exhaustive:
their ranks are `32, 5280, 3520, 1408, 320`, the minimal-polynomial residual
is zero on all 10,560 basis columns, and the projector report passes. The
versioned higher-bidegree inventory binds projector artifact SHA-256
`a616e996fb8b002473743840051df5792dfeef6d5b43c7fe378d8a9d0e2cab6d`,
Cartesian basis hash `f2dfae7e9422a639142622e431fcf10166edeb9ae9f5976169ec638a3148e739`,
Casimir operator hash `50bd1a225f783092467131c525075be88e3f00b89561804305179150a21e421a`,
and projector-polynomial hash
`cd27eea281fe760fe010aeb24d638d50138044c6b52baf55c34c60df51c9ff91`.
A single Cartesian witness row is not itself one canonical irreducible
summand; its five projected components must be evaluated explicitly.
The resulting v3 higher-bidegree inventory has SHA-256
`0e595b3787e9d9c1c60090b270bdc7a967efcea064850d9f3531d103b49bb52f`
and records the certified first `(0,2)` generator by its noncircular semantic
stream hash, exact rank, and zero target-projector residuals.

The pinned `(2,1)` witness is necessarily populated by the raw 52-dimensional
space because the teleparallel target itself is one exact equivariant member
and has coefficient `1/1280` there. None of the four `(0,2)` maps can populate
that row because their momentum degree is two while the row has momentum
degree one.

Factoring ordered derivatives through the five form summands of `S tensor
Hhat` gives 51 formal downstream coefficient columns before antisymmetrizing
the two spinor derivatives. This is not a rank-51 certificate after PBW
antisymmetrization. The antisymmetrizer recouples form and hook intermediate
bases. An exact Racah/highest-weight recoupling matrix is required before
calling any remaining direction exceptional.

## 3. Gauge and compensator meaning

The known quotients sharply reduce the raw inventory.

1. `H_trace = Gamma^a chi` is exactly the 32-dimensional image of the local
   gamma-trace symmetry in Eqs. (2.2)-(2.3). It is not an added physical
   coefficient. A descended physical operator must annihilate it or send it
   into the declared target-gauge image.
2. `Psi_[2]` is the sole inhomogeneous local-Lorentz compensator in the exact
   source-fixed orbit: `delta Psi_[2]=Lambda_[2]`. Consequently
   `p wedge Psi_[2]` is precisely a target `A3` gauge shift and has zero `G4`
   image.
3. `Psi`, `Psi_[1]`, `Psi_[3]`, `Psi_[4]`, and `Psi_[5]` are invariant under
   that local-Lorentz orbit. This does not by itself make every raw map
   physical.
4. In the constrained-frame implementation, `Psi_[1,3,4,5]` are algebraically
   solved from `D H` by Eq. (40). Treating them as independent while also
   retaining their solved copies double-counts the same source directions.
   Only explicitly declared homogeneous solutions may be added.
5. `Psi` is the independent scalar scale/conformal compensator already present
   in `LinearizedFrameSuperfields`. Its complete super-Weyl quotient has not
   been certified, so it must remain a separately tagged source column.

After the gamma-trace quotient, local-Lorentz quotient, and Eq. (40) solve, the
present independent coordinate source is exactly `320 H_hat + 1 scale`. This
explains the existing 321-column operator.

## 4. Consequence for the zero-support teleparallel witness

Adding independent direct-sum source fields cannot cancel a residual on an
existing `H_hat` source column. Source ordinal is part of the operator row
key. Lorentz equivariance also prevents the gamma-trace `S` block from mixing
with the gamma-traceless `T` block.

More strongly, at the relevant one-spinor, one-momentum `H_hat` slice there
are exactly three raw G4 maps and the exact Bianchi kernel is one-dimensional.
That one closed ray is the corrected trace ray already compared against the
teleparallel target. Therefore none of the following repairs the recorded
zero-common-support row on unrestricted `H_hat`:

```text
adding the 32 gamma-trace coordinates as independent columns;
adding the scale column as an independent column;
adding independent Psi_[p] columns;
adding Gamma4 exterior or hook as extra closed H_hat coefficients.
```

Added fields can occupy analogous PBW monomials on their own source columns,
but they cannot alter the failed `H_hat` column. There are only two legitimate
ways to change that conclusion:

1. prove a source constraint that pulls an added field back to a differential
   function of `H_hat`, then compare the pulled-back operator on the same 320
   columns; or
2. move to a genuinely different `H_hat` bidegree and prove that it has the
   same engineering degree and belongs in the physical ansatz.

Eq. (40) already supplies the audited pullbacks of `Psi_[1,3,4,5]`, so merely
reintroducing those fields does not create a new channel.

## 5. Smallest executable oracle

The smallest decisive program should be witness-first and source-tagged.

### Gate A: raw 352 split

Construct one gamma-trace basis vector

```text
H_alpha^a = (Gamma^a)_alpha{}^beta chi_beta
```

for one `chi` basis coordinate. Verify exactly that `P_320 H=0`. Stream it
through the existing `DH/DDH` builder without applying the physical
representative projection. This tests the second ambient A3 ray, but its
result must be tagged as source gauge, never fitted against an `H_hat` row.

### Gate B: scale column

Set `scale=1`, `H=0`, and stream `DScale/DDScale`. Feed `DScale` into the
existing Eq. (25) fermionic frame input. Record its teleparallel `D G4` image,
Bianchi image, and target-gauge image. This closes the one omitted independent
column without recomputing the 320-column H scan.

### Gate C: pulled-back added channels

For every proposed new constraint `L_j: H_hat -> added field`, compute only
the recorded first witness row on source ordinal zero. The current trace-ray
value is zero there and the target value is exactly `1/1280`. Reject any
`L_j` whose candidate value is also zero. Only survivors earn a full
three-prime 320-column solve. This is a tiny coefficient matrix, not a large
GPU problem.

The production anchors are:

```text
src/eleven_dimensional_h_hat_jet.rs:46       canonical P_320 basis
src/eleven_dimensional_h_hat_jet.rs:213      raw DH/DDH/scale jet visitor
src/eleven_dimensional_physical_curvature.rs:1035  Eq. (25), including DScale
src/eleven_dimensional_physical_curvature.rs:2593  Eq. (40) p=1,3,4,5 solve
src/eleven_dimensional_physical_curvature.rs:2634  differentiated Eq. (40) solve
src/eleven_dimensional_corrected_full_chain_oracle.rs:434  corrected form solve
src/eleven_dimensional_corrected_full_chain_oracle.rs:621  teleparallel stream
src/eleven_dimensional_corrected_full_chain_oracle.rs:680  p wedge Psi_[3]
src/eleven_dimensional_corrected_full_chain_oracle.rs:711  corrected paired streams
```

Expected first implementation sizes are `1` gamma-trace canary column, `1`
scale canary column, and at most the number of explicitly proposed pullback
constraints at the single witness row. The full fallback remains 320 source
columns times only the surviving coefficient columns. No dense
`1376 x jet-dimension` matrix should be materialized.

## 6. Boundary

This inventory proves low-bidegree Lorentz multiplicities and identifies the
known quotient directions. It does not prove that the displayed bidegrees are
the complete physical engineering-degree slice, does not invent missing
source constraints, and does not turn an independent compensator column into
a cancellation on `H_hat`.
