# GPU plan for exact teleparallel Lorentz descent

Date: 2026-08-31

## Scope and stop rule

The corrected gauge-fixed teleparallel `D G4` stream is not currently an
intertwiner from

```text
Lambda^2 S tensor V* tensor Hhat -> S tensor Lambda^4 V*
```

on the raw target. The exact witness-source audit finds 1,032 nonzero
commutator coordinates across the 55 Lorentz generators. Therefore the
57-column augmented rank screen is not a physical no-go and must not be rerun
until the commutator vanishes after a proved local-Lorentz or target-gauge
descent.

The GPU track computes and reduces the cocycle

```text
C_ab = K_ab^(target) F - F K_ab^(source)
```

exactly over three pinned finite fields. It does not invent the quotient image.
An explicit image operator with a bound basis and exact provenance is a hard
input.

## Frozen exact actions

The target action is the existing twice-generator

```text
K_ab = Gamma_ab + 2 M_ab
```

on `S tensor Lambda^4 V*`. Each target basis coordinate has one spinor edge
and at most four form edges.

The source action is the same twice-generator on
`Lambda^2 S tensor V* tensor Hhat`:

* `Gamma_ab` on each exterior spinor slot, with slot-aware wedge reorder sign;
* `2 M_ab` on the formal covector momentum;
* the exact gamma-traceless `Hhat` action, including spinor and vector parts.

The source action is accepted only because diagrams 0, 21, and 238, spanning
outer Fierz degrees 0, 3, and 4, each have zero residual for all 55 generators
at the pinned source canary. A mutation that uses the first-slot wedge sign on
the second spinor slot fails.

## Device data

All records are little endian and ABI-pinned.

```text
ModularMapEntry, 32 bytes
  source u32
  target u16
  reserved u16
  residue_re[3] u32
  residue_im[3] u32

TargetActionEdge, 12 bytes
  input_target u16
  output_target u16
  coefficient i16
  generator u8
  reserved[3]

SourceTransposeEdge, 16 bytes
  output_source u32
  input_source u32
  coefficient i16
  generator u8
  reserved u8
  batch_owner u32

CocycleEntry, 32 bytes
  key u64 = ((generator * source_count + source) * 10560 + target)
  residue_re[3] u32
  residue_im[3] u32
```

Target actions are precomputed once. Source transpose actions are generated
per source batch because materializing all 55 actions over 1,745,920 D21
sources would waste memory. Every descriptor blob has a semantic hash and a
raw SHA-256 hash.

## Fused kernel

Process one generator and one bounded Hhat batch at a time.

1. Upload the corrected target COO for the batch as three-prime residues.
2. `emit_target_action` scatters each map entry through target-action CSR.
3. `emit_source_action` joins each map entry against source-transpose CSR and
   emits the negative source term.
4. CUB radix sort orders 64-bit cocycle keys.
5. CUB `ReduceByKey` adds six modular lanes.
6. A flag kernel compacts entries that are nonzero at any pinned prime.
7. Capture counts, first key/value, and a deterministic prime-major digest.
8. If an exact quotient-image CSR is supplied, apply its declared projector or
   retained-pivot reducer before step 6 and record both raw and quotient counts.

No dense `source x target` buffer is permitted. The device output is the
compact residual only.

## Memory and batching

For one Hhat column the corrected RHS has 343,720 exact entries across both
PBW branches. The D21 portion has 342,640 entries. With a conservative ten
emissions per input, one generator needs about 3.5 million temporary entries.
At 40 bytes per sort input/output pair plus CUB workspace this is under 500 MB.
Five generators in flight remain under 3 GB, leaving ample room on a 24 GB
RTX 4090 for action CSR, quotient CSR, and checkpoints.

The complete target has about 180 million nonzero entries. The exhaustive
raw audit performs about 9.9 billion generator-entry visits. These are signed
small-integer scatters and modular additions, not symbolic algebra. Expected
wall time after the first measured production batch is 2 to 8 minutes on the
RTX 4090, with a hard 30-minute launch budget. This is a projection until the
first million-entry benchmark is recorded.

## Checkpoint and observability contract

Checkpoint identity binds:

* target stream manifest and batch hashes;
* lexicographic-to-numeric four-form basis join hash;
* source and target action hashes;
* pinned primes and denominator ledger;
* quotient-image basis and projector hashes, if present;
* CUDA source, host source, executable, GPU model, driver, and toolkit hashes.

Every five seconds publish generator ordinal, Hhat range, input entries,
emitted entries, reduced entries, quotient residual entries, entries per
second, ETA, VRAM high-water mark, and first residual if present. Checkpoints
are written atomically after each generator and Hhat batch. Resume adoption
requires exact manifest identity.

## Validation ladder

1. Target generator: exact CPU/GPU parity on every edge for all 55 generators.
2. Source generator: exact CPU/GPU parity on the pinned source and all of its
   neighbors.
3. Invariant diagrams: diagrams 0, 21, and 238 have zero commutator for all 55
   generators. Wrong second-slot wedge sign must fail.
4. Teleparallel witness: reproduce 1,032 raw residual rows and the exact first
   residual from the CPU artifact.
5. One million synthetic entries: device sort-reduce equals CPU three-prime
   COO and establishes measured throughput.
6. One real Hhat column: all residual keys and residues match CPU.
7. Exhaustive raw cocycle: report per-generator and aggregate counts/hashes.
8. Quotient canary: a known image vector reduces to zero and a one-entry
   mutation survives.
9. Exhaustive quotient cocycle: zero is mandatory before any new coefficient
   solve.
10. Only then rerun the 56-column augmented solve and exact-replay every pivot.

## Measured fixed-source closure

The first production rung is complete on source coordinate `131857`, formal
momentum axis `5`:

* the exact `D Psi_[2]` response matrix has ambient width 1,760, rank 320, and
  a canonical 320-column basis with 38,400 nonzeros;
* the 55 raw Lorentz commutators have 1,032 nonzeros in total;
* the three-prime device solve and all-10,560-row CSR replay return residual
  counts `[0,0,0]`;
* device time is 26.740 ms on the RTX 4090;
* resident memory is 18,359,116 bytes and high-water memory is 19,670,220
  bytes;
* every device coefficient agrees with the exact `Q(i)` reconstruction, whose
  canonical solutions contain 55 nonzero original-column terms in total;
* every exact reconstructed commutator replays with zero residual.

This proves fixed-source vertical-image membership. It does not prove that
declaring a quotient is physically valid.

## Horizontal correction gate

The paper-derived Eq. (25) correction is implemented independently as

```text
Q_a^gamma = -(i/64) (Gamma_a C)^(beta delta)
                    D_beta Psi_de (Gamma^d Gamma^e)_delta^gamma.
```

For one stored increasing two-form coordinate, the ordered Einstein sum gives
the required factor two. The implementation does not call the repository's
`DDelta` injection or Eq. (25) frame helper. On all 1,760 pure `D Psi_[2]`
directions, `raw + Q` has zero potential rows and zero curl/`D G4` rows at all
11 momentum axes. Omitting the ordered-pair factor leaves 211,200 `D G4`
rows, and sign, normalization, time-metric, and spinor-variance mutations are
also nonzero.

The remaining publication gate is stagewise equivariance on the full Hhat
chain. The corrected `DDelta`, horizontal Eq. (25) frame, target curl, and
teleparallel `D G4` stages must be checked with the same source and target
actions for all 55 generators. The first nonzero stage must be localized, and
the final corrected representative must have both zero raw commutator and zero
pure-vertical response before the 56-column coefficient solve is rerun.
