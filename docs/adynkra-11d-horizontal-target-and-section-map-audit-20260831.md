# Horizontal Eq. 25 target and section-map audit

Date: 2026-08-31

## Verdict

The horizontal correction is exact as a transformation-law correction, but it
does not change any target value on the canonical `Psi_[2]=0` section. The
current 56-column consumer serializes the same target rows that produced the
quarantined rank-53 augmented D21 matrix. A new augmented run would therefore
repeat valid linear algebra on the same non-intertwining section values. It
would not yet be a physical no-go.

## Physical interpretation of Eq. 25

hep-th/0101037 defines the vector frame by its Eq. (2). In Eqs. (20)-(23), the
theta-zero spinor coefficient of that vector frame is identified with the
component gravitino. Eq. (25) gives the general linearized coefficient

```text
psi_a{}^gamma = (i/32) (gamma_a)^{beta delta}
                         D_beta Delta_delta{}^gamma
                + (i/32) (D_beta Psi) (gamma_a)^{beta gamma}.
```

The same paper identifies `Psi_[2]` as the local-Lorentz compensator. With

```text
Delta_[2] = (1/2) Psi_de gamma^{de},
```

the canonical horizontal subtraction is

```text
Q_a{}^gamma(D Psi_[2])
  = -(i/64) (gamma_a)^{beta delta}
               (D_beta Psi_de) (gamma^{de})_delta{}^gamma.
```

The increasing-pair repository basis stores one `de` pair. Einstein summation
contains both ordered pairs, which supplies the factor two in the executable
formula.

The exhaustive gate
`paper_horizontal_eq25_correction_kills_all_pure_psi2_directions` checks all
1,760 `D Psi_[2]` columns. It measures 19,360 raw potential rows, zero corrected
potential rows, and zero corrected curl-to-`D G4` rows over all eleven momentum
axes. A half-pair mutation leaves 211,200 rows.

This does not mean that the unrestricted Eq. (25) coefficient is already a
physical component extractor. The map from the 1,760-dimensional first jet of
the super-Lorentz parameter onto the 352-dimensional frame-spinor coefficient
has rank 352. Hence an arbitrary Eq. (25) frame coefficient can be shifted by
the unrestricted first super-Lorentz jet. A physical component gravitino
requires a Wess-Zumino-preserving residual-gauge specification or an independent
component source.

hep-th/0107155 Eq. (3.2f) states `T_ab{}^gamma=C_ab{}^gamma` and
identifies this anholonomy as the teleparallel gravitino field strength. Its
local-Lorentz transformation is homogeneous. Consequently, an arbitrary
isolated `D Psi_[2]` variation with every other frame datum held fixed is not
the physical vertical orbit. The physical orbit is the correlated section
cocycle produced by transforming and resecting the complete frame. Reducing
each target row by the full raw `D Psi_[2]` image would overgauge the problem:
because that image spans all 352 frame-spinor coordinates, it would annihilate
the entire Eq. (25)-derived target.

## Stagewise descent result

The exact stagewise witness at source coordinate 131,857 checks all 55 Lorentz
generators and gives

```text
DDelta residual rows                    240
DDelta rows outside Psi_[2] image       226
Eq. 25 frame residual rows               81
Eq. 25 frame rows outside image           0
horizontal Eq. 25 frame residual rows     0
curl residual rows                       72
horizontal curl residual rows             0
D G4 residual rows                     1,032
horizontal D G4 residual rows             0
image ranks                  1760, 352, 320, 320
```

This certifies cancellation of the generator-dependent commutator by the
section connection. It does not construct a new pointwise target map.

## Current API path

The current APIs are:

* `horizontal_corrected_full_chain_streams` in
  `src/eleven_dimensional_corrected_full_chain_oracle.rs`.
* `teleparallel_rhs_column` in
  `src/eleven_dimensional_four_form_56_physics_rows.rs`.

The wrapper explicitly records

```text
section_psi_two_is_zero = true
section_values_unchanged_by_horizontalization = true
```

The RHS consumer checks those booleans and serializes `section_target`
unchanged. It does not serialize the connection, modify the target rows, or
change the RHS stream digest. Therefore its Bianchi result is unchanged as
well.

Before a horizontal certificate can be bound into a run manifest, the
following immutable items are still required:

1. semantic hash of the Cartesian `Q` operator;
2. artifact containing exhaustive pure-`Psi_[2]` and mutation counts;
3. artifact containing the stagewise 55-generator descent result;
4. hashes of the source, target, gamma, charge, and basis conventions;
5. an explicit statement that ordinary commutator residuals remain nonzero;
6. a solve contract that treats the section connection as an affine gauge
   relation rather than an ordinary Hom-space column.

## Smallest section-map ansatz

Let

```text
X = Lambda^2 S tensor V tensor Hhat,
Y = S tensor Lambda^2 V.
```

Exact B5 character arithmetic gives

```text
Y = 00001 + 01001 + 10001,
mult_X(00001,01001,10001) = (7,14,13),
dim Hom_g(X,Y) = 34.
```

Factoring through the unique inner conventional solve

```text
S tensor Hhat -> Lambda^2 V
```

leaves seven ordered downstream channels, with target multiplicities `(2,2,3)`,
before antisymmetrizing the two derivative-spinor slots.

An invariant map in `Hom_g(X,Y)` cannot cancel a nonzero ordinary commutator,
because its Lie-algebra coboundary is zero. The needed section correction is a
non-equivariant zero-cochain. For semisimple `so(1,10)` and a finite-dimensional
module, Whitehead's lemma gives `H^1=0`, so a genuine connection cocycle is
integrable to a zero-cochain. That zero-cochain is determined only modulo the
34-dimensional invariant Hom space. Whitehead's lemma proves absence of a
cohomological obstruction, not a canonical physical choice.

The finite executable equation is

```text
d(J B) = -dT,
```

where `T` is the gauge-fixed target and `J` is the exact
`D Psi_[2] -> Eq.25 frame -> curl -> D G4` chain. It must be supplemented by a
paper-derived Wess-Zumino or component normalization. Without that extra gate,
choosing `B` equal to the full Eq. (25) frame makes the target zero and exposes
the ambiguity.

The correct solve uses the particular generator-dependent cocycle
`kappa_X(H)`. It must not replace that correlated image with an unrestricted
rank-320 target quotient independently at every source coordinate.

## Correct normalization boundary

The safe physical comparator is an independent gravitino curvature
`C_ab{}^gamma`. At flat four-form background, hep-th/0107155 Eq. (3.1g) fixes

```text
D_alpha G_bcde = -(1/8) (gamma_[bc)_alpha{}^beta C_de]beta.
```

The repository already implements this as
`linearized_gravitino_curl_to_d_f_four_operator`, with its exact left inverse.
The independent-A3 adapter certifies the `A3 -> G4 -> D G4` gauge complex and
this Eq. (3.1g) convention without claiming a map from `Hhat` to the physical
component gravitino.

## Augmented-rank review

For the unchanged section target, the previous modular calculation is internally
sound:

```text
D21 candidate rank       52
D21 augmented rank       53
D02 independent rank      4
global candidate rank    56
global augmented rank    57
```

The ranks agree at all three pinned primes, and all 53 pivot rows are replayed
against exact CPU entries. Thus the target is outside the selected ordinary
Hom-space span as a statement about the serialized gauge-fixed matrix.

That result cannot be promoted to a physical no-go until the target and the 56
candidate columns inhabit the same transformation category. Renaming the
unchanged RHS stream as horizontal does not satisfy that gate.
