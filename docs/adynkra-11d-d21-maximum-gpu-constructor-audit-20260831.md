# D21 maximum-GPU constructor audit

Date: 2026-08-31

## Verdict

The current 400-diagram grammar can be represented by a 6,400-byte immutable
device manifest. Every hot-path Cartesian operation is reducible to table
lookup, signed-permutation composition, integer multiplication, modular sparse
SpMV, and modular elimination. Exact rational pivot replay, grammar generation,
provenance hashing, and certificate publication should remain on CPU.

The epsilon/Hodge obligation is now closed exactly in the native grammar:
the volume element is `-I` with zero residual, all 1,024 rank-zero-through-five
gamma masks satisfy the complementary-mask Hodge identity, and the mutation
gate detects convention drift. Therefore no epsilon opcode is required.

## Frozen descriptor

The canonical host order is Rust derived ordering on
`(r,s,cross,outer_external_indices,inner_external_indices,metric_pairs)`.
External codes are momentum `0`, H vector `1`, and output form slots `2..5`.

Each descriptor is 16 little-endian bytes:

| Bytes | Type | Meaning |
|---:|---|---|
| 0 | `u8` | outer `C Gamma_[r]` degree |
| 1 | `u8` | inner `Gamma_[s]` degree |
| 2 | `u8` | number of normalized cross-gamma contractions |
| 3 | `u8` | six-bit outer external mask |
| 4 | `u8` | six-bit inner external mask |
| 5 | `u8` | canonical metric-pair count |
| 6..7 | `u16` | reserved, required zero |
| 8..11 | `u32` | up to three sorted pairs, six bits per pair |
| 12..13 | `i16` | normalization numerator |
| 14..15 | `u16` | positive normalization denominator |

Each endpoint uses three bits. Pair endpoints and the pair list are sorted.
The current normalization is `1/1`: one canonical antisymmetric cross
contraction per increasing internal-axis combination, with no implicit
`cross!` factor.

Frozen files:

- `results/adynkra_11d_d21_device_diagrams_v1.bin`
- Blob SHA-256: `ecf6545e4b6cc997a9f6ad5744e810892b76837b6eb6dd1c4e8f8f97757c901c`
- Semantic SHA-256: `296705a2da28e4c28db931c909833be0e736ef5da3e0598901afcd87c6d0049c`
- `results/adynkra_11d_d21_device_diagram_manifest.json`
- Sidecar SHA-256: `11380df7c3ff7a1aa24167e9249df1a333df13f0f5220233b5bb4ad03dba6136`
- Native grammar source SHA-256:
  `9701f31ccb7f6b6db080aedb5b36af4da29375d696e415a0757a6a37749ad636`
- Native 32-byte packed semantic SHA-256:
  `d5d6290cdaaa6dfc59c08853b5fda2963f23194099a805d8c7f3d54ccb7bcbe1`

The manifest contains 400 descriptors and reproduces the native grammar
histograms:

```text
outer degree: 0 -> 21, 3 -> 209, 4 -> 170
inner degree: 0 -> 9, 1 -> 48, 2 -> 75,
              3 -> 98, 4 -> 107, 5 -> 63
cross count:  0 -> 101, 1 -> 137, 2 -> 126, 3 -> 36
```

Host validation must reject any descriptor unless:

1. `popcount(outer_mask)=r-cross`.
2. `popcount(inner_mask)=s-cross`.
3. Outer and inner masks are disjoint.
4. Metric endpoints are exactly the complement of their union.
5. Every endpoint occurs once and no output-output metric pair occurs.
6. Pairs are sorted, unused packed bits and reserved bytes are zero.
7. The normalization denominator is positive and admissible at all three
   proof primes.
8. The decoded array re-encodes byte-for-byte to the pinned blob.

## Device tables

The following tables are sufficient for the hot path:

| Table | Compact representation | Approximate size |
|---|---|---:|
| `Gamma_[s]`, `s<=5` | 1,024 signed permutations of 32 lanes | 64 KiB |
| `C Gamma_[r]`, `r=0,3,4` | 496 signed permutations of 32 lanes | 31 KiB |
| Lorentz metric | eleven signed bytes | 11 B |
| Numeric four-form basis | 330 masks plus four axes per mask | 1.9 KiB |
| Canonical Hhat basis | 320 columns, two signed coordinates each | under 4 KiB |
| Cross axes | increasing combinations for counts 0..3 | under 1 KiB |
| Source spinor pairs | 496 increasing pairs | under 2 KiB |
| Casimir C4 | 29 modular entries per target column | about 2.5 MiB per prime |

Gamma and charge-gamma products must not be stored as dense `32x32` matrices.
They are signed permutations. The combined table is too large for 64 KiB
constant memory, so put the small metric/mask/offset tables in constant memory
and the signed permutations in read-only global memory. They should remain L2
resident.

The current native upload API is correct but not yet maximum-density: its
descriptor is 32 bytes and its two flattened gamma tables occupy 8 MiB as
`i16`. Moving to the 16-byte descriptor and signed-permutation tables cuts the
manifest in half and the gamma payload by roughly 84 times. The native query
record is 14 bytes; either mirror it with an explicitly packed CUDA struct or
pad/reorder it to an aligned 16-byte ABI before fleet use.

## GPU-factorable operations

These operations require no CPU fallback:

- descriptor decoding and validation flags;
- external-axis routing;
- Lorentz time-sign lookup;
- metric delta tests;
- repeated-axis rejection and gamma-mask construction;
- inversion parity by bit population count;
- cross-axis combination lookup;
- charge-gamma and gamma signed-permutation lookup;
- two-term Hhat basis expansion;
- modular accumulation and deterministic sort/reduce;
- target Casimir projection as four sparse polynomial SpMV passes;
- modular row sketching and RREF at each proof prime.

Do not materialize the complete `400 x 1,745,920 x 10,560` tensor. Evaluate
canonical actual rows or deterministic linear functionals. A sketch rank is a
lower bound on the original operator rank, so reaching the exact character
upper bounds `[7,7,11,14,13]` is decisive.

## Operations intentionally left on CPU

- generation and canonical sorting of the grammar;
- replay of the already-passing epsilon-to-gamma-Hodge certificate;
- SHA-256 and immutable manifest assembly;
- denominator admissibility and modular inverse preparation;
- exact rational reconstruction of selected pivot entries;
- denominator clearing and Bareiss determinants;
- report-last publication and replay validation.

No evaluated Cartesian contraction is intrinsically CPU-only. Explicit epsilon
vertices are absent because the exact gamma-Hodge certificate proves them
redundant in the fixed 11D convention.

## CPU canaries

### Manifest canary

Decode, validate, re-encode, and hash all 400 descriptors. Require exact count,
all histograms, blob hash, semantic hash, and zero invalid descriptors.

### Seven-diagram operation-cover canary

A deterministic greedy set cover over every outer degree, inner degree, cross
count, pair count, external routing role, and allowed metric-pair type selects
ordinals:

```text
[69, 287, 218, 16, 61, 159, 238]
```

Their signatures are:

```text
69:  (r3,s3,c0; outer M,H,O0; inner O1,O2,O3; no pairs)
287: (r4,s2,c1; outer O1,O2,O3; inner M; pair H-O0)
218: (r3,s5,c3; inner O0,O1; pairs M-O2,H-O3)
16:  (r0,s4,c0; inner H,O0,O1,O2; pair M-O3)
61:  (r3,s1,c1; outer O0,O3; pairs M-O1,H-O2)
159: (r3,s3,c2; outer O2; inner O3; pairs M-O0,H-O1)
238: (r4,s0,c0; outer O0,O1,O2,O3; pair M-H)
```

For each diagram, scan canonical source tuples until the first nonzero CPU
vector is found. Freeze that tuple, nonzero count, exact integer stream hash,
and maximum absolute accumulator. GPU must reproduce the complete vector at
all three primes. Include time and spatial momentum/H witnesses across the
seven records.

### Existing outer-zero canary

Retain the native 21-diagram outer-degree-zero canary. Its single-source ranks
are lower bounds only. CPU and GPU must match its raw stream digests and all
five projected rank results; neither side may label it a completion proof.

## Rank and exact replay gates

For the full run, partition by source Fierz channel and target sector. Required
flattened-operator ranks are:

```text
scalar:  [1,0,0,1,1]
lambda3: [3,2,5,6,6]
lambda4: [3,5,6,7,6]
total:   [7,7,11,14,13] = 52
```

At each prime:

1. Evaluate deterministic actual rows or linear sketches for all 400 columns.
2. Apply the target projector polynomial modularly.
3. RREF in canonical diagram order.
4. Require every channel-sector rank above, not only total rank 52.
5. Record pivot diagram ordinals and actual canonical row keys.
6. Continue actual-row search until a square pivot minor exists for every
   nonzero block.

CPU replay then recomputes every selected entry over `Q`, clears denominators,
and evaluates the determinant with Bareiss elimination. Require:

- nonzero exact determinant for every block;
- determinant residues equal GPU residues at all three primes;
- all 55 Lorentz commutators vanish on every selected generator;
- source Fierz projector and target projector residuals vanish;
- the union contains exactly 52 independent maps.

The exact character inventory is the upper bound. A certified rank of 52 on a
subset of canonical diagrams proves completeness without enumerating or
storing the full operator.

## Required mutations

1. Flip one manifest external-mask bit. Manifest hash or structural validation
   must fail before launch.
2. Drop one outer-degree-four descriptor. Count/histogram/hash must fail.
3. Flip the time metric sign. At least one seven-diagram CPU canary must fail.
4. Use lexicographic rather than numeric four-form masks. Target projector or
   canary must fail.
5. Swap the outer spinor pair without its exterior sign. Fierz canary must
   fail.
6. Omit one cross-axis combination. Cross-count-two or cross-count-three
   canary must fail.
7. Introduce an implicit `cross!` factor. Exact CPU/GPU coefficient replay
   must fail even if modular ranks remain unchanged.
8. Change one Casimir eigenvalue. Projector residual or sector ranks must fail.
9. Corrupt one pivot row key. Exact determinant replay must fail.
10. Mutate one Hhat time component. Gamma-trace or Lorentz-equivariance gate
    must fail.

## Acceptance decision

Launch is acceptable only after the native Rust exporter semantically decodes
to the same 400 signatures as the independent compact freezer and all seven
operation-cover canaries have immutable exact CPU witnesses. Raw byte identity
is not expected while the native exporter retains its 32-byte array descriptor
and the independent freezer uses the 16-byte bit-packed descriptor. The exact
epsilon/Hodge gate is green. Compact descriptor parity and the seven frozen
CPU witnesses remain launch gates.
