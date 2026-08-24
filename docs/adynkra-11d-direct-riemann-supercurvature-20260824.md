# Exact direct frame-to-Riemann supercurvature record

**Date:** 2026-08-24

**Run window:** 2026-08-24 14:09:43 to 14:45:08 MDT
**Status:** Direct off-shell linearized Riemann branch implemented, persisted, adopted, and proved injective on the declared 321-column gauge-fixed source basis. Full physical `F` and physical `K` are not complete.

## 1. Result in one paragraph

The source-fixed bosonic frame in hep-th/0101037 Eq. (25) was composed directly with the repository's exact free-graviton curvature complex. This gives the full off-shell linearized Riemann tensor from the canonical gauge-fixed `H_hat` frame without identifying `D^2 W_[4]` with gravity and without imposing Ricci flatness. The new Riemann sector was merged into the existing 321-column invariant-supercurvature stream under a versioned, fail-closed shard schema. A fresh four-thread run wrote 321 exact shards containing 18,931,260 terms, of which 1,211,340 are direct Riemann terms. The complete declared operator and its Riemann sector separately have rank 321 over `F_p(i)` at `p = 1,073,741,783`. Because `p = 3 mod 4`, this is a field, and modular full column rank proves exact column independence over `Q(i)`. This is an injectivity result for the declared gauge-fixed diagnostic operator. It is not the physical target-gauge quotient and is not a physical-`K` result.

## 2. Scientific construction

### 2.1 Source basis and gauge section

The domain has 321 columns:

1. 320 canonical Cartesian gamma-traceless frame-spinor basis vectors, indexed as ten spatial vector positions times 32 real Majorana spinor positions.
2. One scalar compensator column, ordinal 320 and source name `scale`.

At the typed input boundary the implementation applies

\[
\widehat H = P_{320} H
\]

and clears the independent local Lorentz two-form coordinate. This is a canonical local-Lorentz gauge section. It is not a proof that every raw `J/T/W` coordinate descends unchanged through the local Lorentz orbit. The construction report records that the raw orbit response is nonzero and keeps that distinction fail-closed.

### 2.2 Bounded first-spinor jet

Eq. (25) needs only `D H`, not the complete `D D H` jet. The implementation therefore adds a bounded visitor for

\[
D_\alpha \widehat H_\beta{}^m,
\qquad
32 \times 32 \times 11 = 11,264
\]

possible coordinates. For one stored `H` coordinate it emits exactly 32 first-spinor descendants. A focused equality test compares this visitor entry-for-entry with the `DH` sector of the full ordered jet.

### 2.3 Eq. (25) inverse frame

In the repository's all-real Cartesian Majorana convention, the source-fixed sparse operator implements the `D H` coefficient of hep-th/0101037 Eq. (25) as

\[
f_a{}^m\big|_{D H}
=
\frac{i}{16}(\gamma_a)^{\alpha\beta}
D_\alpha \widehat H_\beta{}^m.
\]

The scalar column contributes

\[
f_a{}^m\big|_{\Psi}=\Psi\,\delta_a{}^m.
\]

The independent Lorentz two-form contribution printed in Eq. (25) is absent only because the canonical representative has already fixed that orbit coordinate to zero. The result is an 11 by 11 inverse-frame perturbation, hence 121 frame coordinates.

### 2.4 Inverse frame to covariant metric

Write

\[
E_a{}^m=\delta_a{}^m+f_a{}^m,
\qquad
e_m{}^a=\delta_m{}^a-f_m{}^a+O(f^2),
\]

with mostly-plus metric

\[
\eta=\operatorname{diag}(-,+,\ldots,+).
\]

Then

\[
g_{mn}=\eta_{mn}+h_{mn},
\qquad
h_{mn}
=-\eta_{nc}f_m{}^c-\eta_{mc}f_n{}^c.
\]

This produces the 66 symmetric metric coordinates. Exact tests verify the time-sign convention and show that both a spatial infinitesimal rotation and a Lorentz boost vanish under this symmetrization. These are sensitive tests of the mostly-plus lowering signs, not merely dimension checks.

### 2.5 Metric to full linearized Riemann tensor

The adapter reuses `target_sector_complex(TargetSector::Graviton).curvature`. In the target complex's unit-weight convention,

\[
R_{ab\mid cd}(p)
=p_a p_c h_{bd}
-p_a p_d h_{bc}
-p_b p_c h_{ad}
+p_b p_d h_{ac}.
\]

No additional textbook factor of `1/2` is inserted. The target coordinate basis uses two lexicographically ordered antisymmetric pairs:

\[
\binom{11}{2}=55,
\qquad
55\times55=3,025
\]

ambient pair-pair coordinates. Pair exchange and the algebraic Bianchi identity reduce this to

\[
\dim \operatorname{Riem}(11)
=\frac{11^2(11^2-1)}{12}
=1,210.
\]

The implementation keeps this full algebraic Riemann target type. It does not project to the 1,144-dimensional Weyl representation. The 321-column image is not asserted to span the 1,210-dimensional algebraic Riemann space. The scale-column test has a nonzero scalar-curvature trace, providing an exact executable witness that the Ricci and scalar pieces were not silently discarded.

Every source ordered-superderivative monomial retains its 32-bit exterior-spinor mask. The curvature operator multiplies exactly two additional formal momentum variables into the eleven-component exponent vector. It does not alter the exterior mask.

### 2.6 Why `D^2 W` is not the Riemann definition

The source literature states that successive components of the four-form Weyl superfield contain the four-form, Weyl-gravitino, and Weyl-tensor data, but the audited sources do not print a complete convention-fixed second-descendant projector suitable for this calculation. In particular, hep-th/0107155 Eq. (3.1g) fixes the first-descendant four-form/gravitino relation, while its superspace Bianchi relations show why later descendants mix curvature with derivative-four-form terms. Therefore:

- direct Eq. (25) frame curvature is the production Riemann definition;
- raw `W_2021` remains a separately tagged superfield sector;
- raw two-exterior-D `W_2021` terms are never relabeled or emitted a second time as gravity;
- a future `D^2 W` calculation may be used only as a conditional Weyl cross-check.

This separation prevents an unsupported normalization guess and prevents double counting the gravitational target.

## 3. Unified exact output and shard contract

### 3.1 Output sectors

The production row sectors and stable byte tags are:

| Tag | Sector | Meaning |
|---:|---|---|
| 0 | `XTwo` | exact `(11000)` conventional quotient |
| 1 | `XFive` | exact `(10002)` conventional quotient |
| 5 | `JMinus` | source-fixed invariant `J^(-)` coordinate |
| 8 | `W2021Raw` | raw 2021-convention W superfield coordinate at its preserved exterior-D degree |
| 9 | `LinearizedRiemann` | independent direct off-shell frame curvature |

Rows are strictly ordered by

```text
(exterior_spinor_mask, [p_0,...,p_10] exponents, sector_tag, coordinate)
```

The raw invariant stream and the direct Riemann stream are merged in this order before either hashing or persistence. Both the digest-only path and the shard writer call the same unified visitor.

### 3.2 Versioned schemas

```text
operator:       adynkra-11d-gauge-fixed-invariant-supercurvature-operator-v3
unified output: adynkra-11d-gauge-fixed-invariant-supercurvature-output-v1
column shard:   adynkra-11d-gauge-fixed-invariant-supercurvature-column-shard-v2
magic:          AD11FINVCOL2\0\0\0\0
```

The v2 shard format is intentionally incompatible with v1. It cannot silently adopt a v1 shard.

Each exact entry occupies 67 bytes:

| Field | Encoding | Bytes |
|---|---|---:|
| sector tag | `u8` | 1 |
| output coordinate | little-endian `u64` | 8 |
| exterior-spinor mask | little-endian `u32` | 4 |
| eleven momentum exponents | 11 little-endian `u16` values | 22 |
| real numerator and denominator | 2 little-endian `i64` values | 16 |
| imaginary numerator and denominator | 2 little-endian `i64` values | 16 |
| **Total** |  | **67** |

The validator checks magic, source ordinal and name, complete entry boundaries, allowed tags, strict row order, positive denominators, entry count, semantic SHA-256, and whole-file SHA-256. Shards are written through a temporary file, synchronized, and atomically renamed.

### 3.3 Backward compatibility of the scientific payload

Although the binary schema is deliberately incompatible, the pre-Riemann scientific payload is unchanged. The recorded legacy projection validation filtered the new stream to tags `0,1,5,8` and compared all 321 columns entry-for-entry with the previous certified payload:

```text
columns compared:                 321
legacy projection byte-identical: true
legacy terms:                     17,719,920
new terms:                        18,931,260
new Riemann terms:                 1,211,340
legacy payload SHA-256:           a49b0323885b42f8b0706a778bdee4fdf2554e67d834de38f41dc6f0efcae51d
```

Term counts by sector are:

| Sector | Exact terms |
|---|---:|
| `XTwo` | 112,960 |
| `XFive` | 1,384,320 |
| `JMinus` | 153,168 |
| `W2021Raw` | 16,069,472 |
| `LinearizedRiemann` | 1,211,340 |
| **Total** | **18,931,260** |

## 4. Fresh production run

### 4.1 Provenance manifest

The authoritative provenance directory is:

```text
results/adynkra_11d_complete_f_v3_run_20260824T1409MDT/
```

The manifest records:

| Item | Value |
|---|---|
| Started | `2026-08-24T14:09:43-06:00` |
| Finished | `2026-08-24T14:45:08-06:00` |
| Git HEAD | `586862e19c8c0e433ec451b0ad15deefe503f000` |
| Git diff SHA-256 | `1684d34b28b28d14f6b7ac66fec0066a99d5f6f9aa2b516d2a56d565f78a37ec` |
| Rust compiler | `rustc 1.94.1 (e408947bf 2026-03-25)` |
| Writer binary SHA-256 | `f5cebf2e2ba4e009cc2fb6efc5f4c240a3f65aab651ab4671f6bc85b162ff9cb` |
| Worker threads | 4 |
| Exit code | 0 |

The diff hash is essential because this run was made from the recorded HEAD plus uncommitted scientific changes, not from the HEAD commit alone. It authenticates the historical tree state but is not a standalone reconstruction patch.

### 4.2 Generation and adoption

Fresh generation completed all 321 columns:

| Statistic | Value |
|---|---:|
| Fresh shards | 321 of 321 |
| Exact terms | 18,931,260 |
| Wall time | 2,124.24 s, or 35 min 24.24 s |
| User CPU time | 7,805.09 s |
| System CPU time | 610.91 s |
| Maximum resident set size reported by `/usr/bin/time -l` | 12,791,595,008 bytes |
| Peak memory footprint reported by `/usr/bin/time -l` | 11,720,010,888 bytes |
| Aggregate shard bytes | 1,268,424,189 bytes, 1.1813 GiB |
| Smallest shard | 429,681 bytes |
| Largest shard | 4,031,550 bytes |

The first-pass certificate marked all 321 shards fresh. The immediate adoption pass reopened and fully validated all 321 existing shards in 2.18 s, marking all of them reused while reproducing the same semantic operator hash and term count.

The persisted column-0 `H/raw-W` canary passed 1 of 1 in 18.77 s with peak RSS 2,769,403,904 bytes. This checks a physical frame column, not only the much smaller scale column.

The per-column exact-term histogram is:

| Terms per column | Number of columns |
|---:|---:|
| 6,412 | 1 |
| 54,079 | 16 |
| 54,844 | 16 |
| 55,572 | 32 |
| 60,171 | 256 |

The 6,412-term column is the scalar source at ordinal 320. Its semantic column SHA-256 is `170c795ad52916ec23d3ce85aabf9ebf3bcc58ac3c81938404dc88c38fead4a9`; its shard SHA-256 is `9d06c98065e5125c207f28b2f093228bc58263391392a19c4aa510562069a7ee`.

## 5. Exact rank certificates

### 5.1 Combined invariant-supercurvature operator

Artifact:

```text
results/adynkra_11d_gauge_fixed_invariant_supercurvature_kernel_v2_p0_20260824.json
```

Recorded result:

| Statistic | Value |
|---|---:|
| Fully validated shards | 321 |
| Source columns | 321 |
| Prime | 1,073,741,783 |
| Exact input terms | 18,931,260 |
| Rows examined before full-rank stop | 1,058 |
| Shard entries examined by elimination | 3,794 |
| Rank over `F_p(i)` | 321 |
| Nullity upper bound | 0 |
| Full column rank | true |
| Stopped after full rank | true |
| Certificate elapsed time | 8.440088541 s |
| Process wall time | 8.46 s |
| Peak memory footprint | 329,548,544 bytes |

All shard bytes and footers were validated before the early-stop elimination. The small number of rows and entries examined is therefore an elimination optimization, not partial shard validation.

### 5.2 Riemann-only projection

Artifact:

```text
results/adynkra_11d_gauge_fixed_invariant_supercurvature_riemann_sector_kernel_v1_p0_20260824.json
```

Recorded result:

| Statistic | Value |
|---|---:|
| Sector tag | 9 |
| Sector | `LinearizedRiemann` |
| Fully validated shards | 321 |
| Source columns | 321 |
| Exact Riemann terms | 1,211,340 |
| Rows examined before full-rank stop | 181,895 |
| Riemann entries examined by elimination | 568,907 |
| Rank over `F_p(i)` | 321 |
| Nullity upper bound | 0 |
| Full column rank | true |
| Stopped after full rank | true |
| Certificate elapsed time | 9.562700084 s |
| Process wall time | 9.59 s |
| Peak memory footprint | 305,742,592 bytes |

The Riemann sector alone separates all 321 declared gauge-fixed source columns as polynomial-valued vectors over `Q(i)`. Equivalently, no nonzero constant `Q(i)` linear combination of these 321 source basis columns vanishes identically in the direct Riemann output before imposing a physical target quotient. This does not rule out syzygies with polynomial coefficients.

### 5.3 Exact meaning of the modular result

Both computations use the Gaussian extension at

\[
p=1,073,741,783\equiv3\pmod4.
\]

Thus `x^2+1` is irreducible modulo `p`, and the arithmetic is over the field `F_p(i)`. A nonzero 321 by 321 minor modulo this good prime implies that the corresponding exact minor over `Q(i)` is nonzero. Since the source has only 321 columns, modular rank 321 proves exact rank 321 and exact nullity zero for the declared operator or declared sector projection.

The converse would not hold: modular rank deficiency by itself would not prove an exact kernel. That asymmetric proof boundary is recorded in both JSON certificates.

Final-code reruns of both the combined and Riemann-only kernel commands reproduced every semantic field. Only `elapsed_seconds` changed.

## 6. Validation suite

The implementation was exercised by exact tests at the following boundaries:

1. The bounded first-`D H` visitor exactly matches the full jet's `DH` sector.
2. Mostly-plus time lowering has the expected opposite signs for `f_0{}^i` and `f_i{}^0`.
3. Spatial rotations and Lorentz boosts vanish in the symmetric metric.
4. Riemann pair exchange holds entrywise.
5. The algebraic Bianchi identity holds exactly.
6. The independent target differential-Bianchi operator annihilates the direct curvature at generic formal momentum.
7. A one-coefficient curvature mutation produces a nonzero Bianchi residual.
8. The scale route adds exactly two momentum powers, has a nonzero scalar trace, and therefore is not a Weyl-only projection.
9. The unified stream is strictly ordered.
10. Exterior-D masks survive the 67-byte encoding exactly.
11. V1 magic is rejected, out-of-order v2 payloads are rejected, and nonpositive rational denominators are rejected.
12. Independent digest and writer passes for source ordinal 320 produce the same semantic SHA and term count; reopening the shard reproduces its semantic SHA, file SHA, and byte count.
13. The new stream projected to legacy tags is byte-identical across all 321 columns.
14. The hardened kernel reader suite passes 5 of 5 tests.

Recorded test executions:

| Command or focused gate | Result |
|---|---|
| `cargo test --quiet eleven_dimensional_complete_f::tests` | 16 passed, 0 failed, 2 ignored, 224.30 s |
| `cargo test --quiet eleven_dimensional_h_hat_jet::tests` | 6 passed, 0 failed, 2.53 s |
| Unified scale strict-order and Bianchi gate | 1 passed, 12.93 s |
| Digest, writer, and resume agreement gate | 1 passed, 23.45 s |
| V2 schema and backward-incompatibility gate | 1 passed, 0.00 s |
| Hardened kernel suite | 5 passed, 0 failed |
| Final `cargo check` | passed |
| Final release build | passed |

The two ignored complete-F tests are deliberately expensive full-column persistence/digest tests. Their relevant production behavior is covered here by the persisted canary, fresh 321-column generation, adoption pass, and independent digest-writer-resume gate.

## 7. Artifact and hash ledger

| Artifact | SHA-256 or semantic hash |
|---|---|
| `results/adynkra_11d_gauge_fixed_invariant_supercurvature_operator_v3_riemann_20260824.json` | file: `3f75b3fbf3ae30b494d6918e091c3244e7c304b338e598b20ba2a7cbb8e9598c` |
| same operator | semantic operator: `fec4195161bcd085d629c85bc2b0efc12bc2ea613266c4d3c12c49be37a41661` |
| `results/adynkra_11d_gauge_fixed_invariant_supercurvature_kernel_v2_p0_20260824.json` | `6c2e802a039daa1fd38676051ee4aff0e04e36978426f7e5f8d89d77ca370f93` |
| `results/adynkra_11d_gauge_fixed_invariant_supercurvature_riemann_sector_kernel_v1_p0_20260824.json` | `c68a81f25d3552ea4da6c031a98958445880d173da4d3bc27ae86138126e8b88` |
| `results/adynkra_11d_complete_physical_f_construction.json` | `576d32ed798151cadcb8c278f05f72c6683463656c59c3d421b7ed9bf82b30b6` |
| `results/adynkra_11d_complete_f_v3_run_20260824T1409MDT/legacy-projection-validation.json` | `caa803976846c4e52f68f92f9557bba3bab690b805f0074142b9bc4ff7dd3777` |
| `results/adynkra_11d_complete_f_v3_run_20260824T1409MDT/final-validation.json` | `5c958c42b752fc0b0b9c77163b51d9f177ac788140b19013332e45bb98cf947e` |
| run manifest | `3ee056dc1ab85a27d1ae01b3609d2830e41b190826cdab6029f59f24ca3d65dd` |
| first-pass certificate | `cd84182449425f9cdfb9471097d1b6028fde6dd35702afd1834610cbd7f7f9b5` |
| adopted certificate and public operator JSON | `3f75b3fbf3ae30b494d6918e091c3244e7c304b338e598b20ba2a7cbb8e9598c` |

The first-pass and adopted certificate files differ in operational `shard_reused` metadata. They have identical column semantics, total term count, and semantic operator SHA-256.

The compact committed certificates cryptographically bind every raw shard but do not contain the 1.181 GiB payload. The 321 raw shards remain preserved locally, so independent byte-level replay requires access to that local corpus.

The provenance directory retains the raw machine-readable streams:

| Log | Content |
|---|---|
| `stderr.log` | 321 fresh `column_complete` events and `/usr/bin/time -l` statistics |
| `adoption-stderr.log` | 321 reused-column validation events and adoption timing |
| `kernel-stderr.log` | combined shard-validation and rank progress events |
| `riemann-sector-kernel-stderr.log` | tag-9 shard-validation and rank progress events |
| `complete-f-build-final-stdout.log` | final v6 fail-closed construction report |
| `legacy-projection-validation.json` | 321-column byte-identity check against the legacy payload |

Primary source hashes carried by the final construction report:

| Source | SHA-256 |
|---|---|
| hep-th/0101037 | `9405ca44a0036567cf86bfbc89de097d8b064612c314b28f31d614e4553a4453` |
| arXiv:2007.05097 | `3a6e81c2c677cf3b68455615145510a4d8bce7db967c77c4afd3b85423535df7` |
| hep-th/0107155 | `71ccd43c2dea3df8fb9708c016595463cca2674bccad1872c955fc2c8647f25e` |

The third source pins the first-descendant four-form/gravitino relation that remains conditional. Its inclusion in provenance does not change that proof boundary.

The final v6 construction report deliberately records the following mixed state:

| Report field | Value |
|---|---|
| `direct_off_shell_frame_to_riemann_adapter_implemented` | true |
| `direct_riemann_integrated_into_321_column_operator` | true |
| `direct_riemann_bianchi_certified` | true |
| `theta_two_w_gravity_double_emitted` | false |
| `target_curvature_adapter_implemented` | false |
| `target_bianchi_euler_noether_composition_certified` | false |
| `complete_physical_f_implemented` | false |
| `complete_f_operator_sha256` | null |
| `exact_polynomial_target_kernel_derived` | false |
| `pointwise_or_bounded_kernel_is_accepted_as_physical_k` | false |

The deterministic geometry probe in the same report contains 13 `X_[2]`, 182 `X_[5]`, one each of `J^(1)`, `J^(2)`, `J^(+)`, and `J^(-)`, 265 mixed-torsion, 30 `W_2001`, and 44 `W_2021` nonzero entries. These are smoke-test entry counts, not representation ranks.

## 8. What is proved

The present artifacts prove all of the following within the declared linearized conventions:

- the exact Eq. (25) `D H` frame map and scalar diagonal compose to the symmetric metric;
- the direct target curvature lands in the full off-shell algebraic Riemann representation, not only Weyl;
- algebraic and differential Bianchi identities hold exactly;
- the direct branch preserves exterior-D masks and adds two formal momenta;
- the new Riemann payload was added without changing any legacy invariant payload byte;
- all 321 fresh shards validate and are resumable;
- the complete declared invariant operator has exact column rank 321;
- its direct Riemann projection alone has exact column rank 321.

## 9. What is not proved

The following claims are explicitly not established:

- `complete_physical_f_implemented` remains false;
- `target_curvature_adapter_implemented` remains false for the complete multiplet;
- the full target Bianchi, Euler, and Noether composition has not yet been certified for the assembled physical map;
- the first-descendant `D W_[4]` to gravitino-curl adapter remains conditional on the convention identifying the raw W lowest component with the physical four-form;
- no complete `D^2 W_[4]` component projector has been derived;
- no target physical-gauge quotient has yet been applied to the 321-column response;
- no exact polynomial physical `K` has been derived;
- neither rank certificate is a statement that the Einstein-equation extension exists or is unique;
- this is linearized formal-momentum geometry, not nonlinear eleven-dimensional supergravity closure.

In particular, Riemann-only full rank before quotient does not imply that the eventual physical quotient has no kernel. Gauge-trivial target responses can become zero only after the correct target gauge image is constructed and divided out.

## 10. Next steps toward physical `K`

1. **Integrate the conditional first descendant.** Route complete fixed-momentum one-spinor `W_[4]` descendants through the exact aggregate left inverse into the 1,760-component gravitino curl, retain the forward image check, and keep the convention gate explicit.
2. **Complete target identities.** Assemble Riemann, four-form, and gravitino-curl outputs under the independent target Bianchi, curvature-to-Euler, Euler-Lagrange, and Noether maps.
3. **Construct the physical target gauge quotient.** This is the decisive semantic step. The present unquotiented injectivity result cannot substitute for it.
4. **Regenerate and hash complete physical `F`.** Only after every target sector and identity is integrated should `physical_target_component_adapter_complete`, `complete_physical_f_implemented`, and the complete operator SHA field become true/non-null.
5. **Compose all 77 exact embeddings.** Evaluate the required `F A G_p` branches, including both the `D^17 Lambda` and `p D^15 Lambda` structures, over generic formal eleven-dimensional momentum and modulo the physical target gauge image.
6. **Solve the exact joint kernel.** A surviving parameter vector is a candidate extension. A zero kernel is an exact obstruction for the declared ansatz.
7. **Verify the equation complex.** Any survivor must map into the exact graviton, three-form, and gravitino complex with physical `44+84|128` content and with curvature, Bianchi, Euler, and Noether compositions vanishing.
8. **Use `D^2 W` only as a cross-check.** If a convention-fixed projector is later derived, compare its Weyl projection with the direct Riemann branch after removing Ricci and scalar pieces. Do not replace the off-shell direct definition with it.

The final construction report correctly leaves `complete_f_operator_sha256` null and `exact_polynomial_target_kernel_derived` false. The next calculation is not another unquotiented rank test. It is the physical target quotient followed by the exact joint-kernel computation.
