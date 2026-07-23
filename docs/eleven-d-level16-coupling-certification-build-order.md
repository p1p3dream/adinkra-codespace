# 11D level-16 coupling certification: build order

Replace the HashMap exterior-algebra engine that failed on the level-16
`wedge^16 S` couplings with an exact, dense/CSR, checkpointed engine that
certifies each source-to-target coupling `V_lambda tensor S -> (10001)` and
writes an atomic per-coupling certificate.

so(11) = B5. `Weight = [i8; 5]`, five simple roots, spinor `S` is 32
dimensional. The scalar 11D superfield's level-16 space is `wedge^16 S`.

## What "done" means

An exact mathematical certificate, not a fast approximation. Every accepted
coupling has: an exact integer/rational primitive coupling and, for each
embedded copy, a residual that is provably zero under all five simple-root
raising operators. A floating-point spectral gap is not acceptable as the
certificate (that is the audit's MINOR-1; this work is what closes it).

## The work list (fixed by the committed fixtures)

8 distinct source irreps, 12 embedded copies. The copy counts are the `_N`
suffixes on the committed kernels in `data/eleven_dimensional_spinor_bridge/`:

| irrep   | copies | fixtures |
|---------|--------|----------|
| (10000) | 1 | `level16_10000_highest_weight_kernel.i16le` |
| (20000) | 1 | `level16_20000_highest_weight_kernel.i16le` (golden, see below) |
| (00100) | 2 | `level16_00100_highest_weight_kernel_{1,2}.i16le` |
| (00010) | 2 | `level16_00010_highest_weight_kernel_{1,2}.i16le` |
| (00002) | 1 | `level16_00002_highest_weight_kernel.i16le` |
| (10100) | 1 | `level16_10100_highest_weight_kernel.i16le` |
| (10010) | 1 | `level16_10010_highest_weight_kernel.i16le` |
| (10002) | 3 | `level16_10002_highest_weight_kernel_{1,2,3}.i16le` |

Total 12 copies. Emit this table as a JSON manifest so the CLI and tests
agree on the work list.

## Current state to build on (do not reinvent)

- Exact kernels are stored as little-endian int16 fixtures in
  `data/eleven_dimensional_spinor_bridge/*.i16le`, embedded via `include_bytes!`
  in `src/eleven_dimensional_spinor_bridge_kernels.rs`. (Note: the level-15
  bridge under `data/eleven_dimensional_bridge/` + `eleven_dimensional_bridge.rs`
  is a different, already-finished computation. Do not confuse the two.)
- Level-15/17 spinor-bridge machinery already works exactly:
  `cmd_adynkra_11d_spinor_bridge_verify`, `cmd_adynkra_11d_spinor_kernel_verify`
  in `src/main.rs`, backed by exact `Ratio<i64>` vectors and `i128` residual
  maps. The new engine must match this rigor, not lower it.
- Existing result JSON is written by shell redirection, which is NOT atomic.
  Do not describe the current artifacts as crash-safe. Atomicity is a new
  property this work adds (Phase 4 / the stonkbot runner), state it as such.

## Non-negotiable invariants (hold these the whole way)

1. **Exactness or it is not a certificate.** A single-prime modular
   "residual = 0" is probabilistic. The residual-is-zero check must be exact:
   exact integer accumulation in a type proven wide enough (checked i128, else
   BigInt), or multi-prime CRT with enough primes to be a proof of exact zero
   for coefficients of the bounded height these systems produce (reconstruct or
   bound, do not trust one prime).
2. **The golden fixture is the committed (20000) coupling at `89f20fc`**
   ("Construct the second leading spinor coupling"). Before any new coupling is
   trusted, the new dense/CSR engine must reproduce that coupling exactly
   (bit-identical kernel and zero residual). Wire it as a test that fails
   loudly. Do NOT use the (00100) coupling as the reference: it is
   experimentally validated but not yet checkpointed or committed (see 0.3).
3. **The abstract coupling is shared across copies; the embedding is not.**
   The per-copy application runs the shared coupling through each copy's own
   embedding into `wedge^16 S`. That embedding is where a subtle bug hides.
4. **Checkpoints are atomic and a late failure cannot erase earlier results.**
   Each accepted coupling is written by temp-file-plus-validate-plus-rename, not
   shell redirection.
5. **Never claim more than the computation proves.** These are couplings into
   `(10001)`; certifying them completes a rigorous result about that channel,
   it does not solve the Gates-Hu open problem. Keep the doc language hedged
   exactly as `docs/adynkra-11d-level15-bridge.md` already does.

## Phase 0: pre-checks (do before writing the engine)

0.1 **Confirm the work list** above against the fixtures on disk and emit the
   JSON manifest (irrep, copy count, fixture filename).

0.2 **Multiplicity check (decides the whole design).** For each of the 8
   irreps compute `dim Hom(V_lambda tensor S, (10001))`, the multiplicity of
   `(10001)` in `V_lambda tensor S`.
   - If every multiplicity is 1: the primitive coupling is unique up to scale
     and the rest of this plan applies directly.
   - If any multiplicity is > 1: STOP and fix a basis of the multiplicity
     space first, and thread that basis consistently into every embedded copy.
     A non-unique coupling applied inconsistently across copies is a
     correctness bug, not a perf bug. (Watch (10002) with its 3 copies and the
     two doubled irreps here.)

0.3 **Freeze the golden fixture; freeze the validated one.** The (20000)
   coupling at `89f20fc` is the regression fixture. Separately, checkpoint and
   commit the experimentally validated (00100) coupling so it stops being an
   uncommitted result; until its artifact is frozen, refer to it only as "the
   experimentally validated (00100) coupling," never as "done."

0.4 **Confirm the diagnosis holds.** Re-profile one representative failing
coupling to confirm time is dominated by HashMap hashing/alloc/merge (the
stated ~95% dense kernel, e.g. 410,860 nonzero of 431,724). Record the
numbers as the justification for the dense rewrite.

**Phase 0 measurement recorded 2026-07-23.** The first `(00100)` fixture has
410,860 nonzero coefficients in a 431,724-column source weight space
(95.17 percent). The three-coupling verifier, including the previously
committed source-kernel checks, completed in 42.83 seconds on the development
machine. The attempted all-coupling HashMap batch was stopped after 1 hour,
42 minutes, 45 seconds with no artifact because it serialized only after every
coupling completed. It remained at approximately one full CPU core and reached
an observed resident-memory peak of approximately 3.1 GB. Process samples
placed the active work in HashMap hashing, allocation, growth, and merge paths.

## Phase 1: abstract per-irrep coupling solver (small, exact)

Solve the representation theory in the small weight space (tens of
dimensions), separated entirely from the exterior embedding.

**Basis method (specify it; this is the hard part).** Do not leave the weight
space basis as "PBW lowering words + Chevalley relations." Fix a canonical PBW
monomial basis and resolve linear dependence with the **Shapovalov Gram
matrix** on each weight space (or an equivalent canonical tensor construction).
The Gram matrix gives an exact, basis-independent handle on which lowering
words are independent and on the inner-product structure used to extract the
primitive coupling. Compute it over the rationals; its rank certifies the
weight-space dimension.

For each of the 8 distinct irreps:
1. Generate the required source weights of `V_lambda` abstractly.
2. Build the five B5 raising matrices on `V_lambda` in the canonical PBW basis
   (reuse the so(11) machinery in `src/eleven_dimensional_prepotential.rs`).
3. Tensor with the 32-dim spinor action `S`.
4. Restrict to total weight `(10001)`.
5. Solve the small exact integer kernel using the Shapovalov-resolved basis
   (primitive integer coupling vector).
6. **Storage: measure, then choose.** Normalize the abstract coefficients,
   measure their height bound, then select checked i16, i64, rational, or
   BigInt storage accordingly (do not assume i16). Confirm the choice against
   the committed `.i16le` fixtures where they already fit; document any that do
   not.

Deliverable: per-irrep abstract couplings + a unit test per irrep that the
kernel satisfies all five raising equations exactly, agreeing with the
committed fixtures.

## Phase 2: dense/CSR exterior action engine (replace the HashMap)

Replace the HashMap term storage with dense/CSR:
- sorted mask tables for the exterior basis of the relevant weight spaces;
- contiguous `Vec` coefficient arrays in the storage type chosen in Phase 1.6;
- precomputed CSR raising and lowering operators for the five simple roots;
- linear accumulation into destination arrays (no per-term hashing);
- **accumulation in checked i128 or BigInt after a documented height bound**,
  so an overflow is impossible rather than merely unlikely.

**Gate (invariant #2):** the first thing this engine does is reproduce the
committed (20000) coupling at `89f20fc` exactly. Only after that passes does it
touch a new coupling.

## Phase 3: per-copy streaming certification (bounded memory)

For each embedded copy of a distinct irrep:
1. Apply only the PBW words its abstract coupling actually uses.
2. Accumulate one simple-root residual at a time.
3. Confirm every destination coefficient is exactly zero.
4. Release that root's working memory before the next root.
5. Write the copy's result immediately (Phase 4).

Never hold all five raising outputs or all couplings resident at once.

## Phase 4: CLI, atomic checkpoints, resume

Independent, resumable subcommands in `src/main.rs`, matching the existing
`adynkra-11d-*` style:
```
coupling-build   --label 00100            # abstract coupling for one irrep
coupling-verify  --label 00100 --copy 1   # certify one embedded copy
coupling-verify  --label 00100 --copy 2
coupling-verify  --all --resume           # skip already-certified copies
```
Each successful coupling/copy writes an atomic
`results/adynkra_11d_level16_coupling_<label>_copy<n>.json` via
**temp-file, validate contents, fsync, atomic rename**, not shell redirection.
`--resume` reads existing artifacts and skips them. A later failure never
erases an earlier certificate.

## Phase 5: parallelize and run on stonkbot (only after 1-4 are correct)

Execution environment: stonkbot (x86_64 Linux, 32 CPU cores, 62 GB RAM).
- **Clean clone** of the branch on stonkbot (fresh, not the dev tree), with a
  sync + build script (`cargo build --release`).
- `rayon` across the 8 distinct irreps and independent copies, with
  **memory-capped worker count** (each copy's dense arrays are large; cap
  workers by a RAM budget, not by all 32 cores blindly).
- The **atomic runner** writes each certificate with temp-file + rename so a
  killed job leaves no half artifact and `--resume` is safe.
- **Local/remote SHA-256 comparison**: after the run, SHA-256 every
  `results/*.json` on stonkbot and compare to a local regeneration to confirm
  the certificates are reproducible and machine-independent.
Do not parallelize before the dense engine is correct and the (20000) fixture
gate passes; a fast wrong answer is worse than a slow right one here.

## Phase 6 (optional, later): multi-prime CRT and GPU

Only once the CPU exact engine is the certificate of record:
- Run the residual check under several primes as independent confirmation, or
  as the primary path with rational reconstruction if BigInts get expensive.
- The per-prime dense modular matrix passes are genuinely GPU-amenable; this is
  the only point where stonkbot's 4090 earns its keep. Not before. A GPU cannot
  rescue the HashMap design and is not on the critical path.

## Acceptance criteria (all must hold)

- `cargo build --release` and `cargo test` green; new per-irrep and
  (20000)-fixture-reproduction tests included and able to fail (add a
  sign-mutation crosscheck like `bbbm_sixteen_onshell_crosscheck.rs` so the
  suite can tell right from wrong).
- All 12 copies certified with exact zero residual, each with an atomic JSON
  artifact; local and stonkbot SHA-256 of every artifact match.
- The (00100) coupling is checkpointed and committed (no longer an uncommitted
  validated result).
- Runtime target met: abstract solve seconds per irrep, per-copy certification
  seconds to a few minutes, all 12 in minutes, bounded memory, visible
  per-coupling progress.
- A short doc (`docs/adynkra-11d-level16-couplings.md`) stating exactly what is
  certified and, in the same breath, what is not (channel-level result, not the
  open problem), retiring the audit's MINOR-1 by pointing at the new exact
  kernels.

## What NOT to do

- Do not make a single-prime modular check the certificate.
- Do not parallelize before the dense engine reproduces the (20000) fixture.
- Do not resolve the abstract coupling per copy.
- Do not assume i16 storage; measure the height bound and choose.
- Do not describe shell-redirected JSON as atomic.
- Do not call the (00100) coupling "done" until its artifact is frozen.
- Do not upgrade the prose ("resolved", "closed") beyond "certified couplings
  into the (10001) channel under the stated setup".

## First move

Phase 0 in full (work-list manifest + multiplicity check + freeze the (20000)
golden fixture and checkpoint the (00100) one + reprofile), then Phase 1 for
one irrep with the Shapovalov-resolved basis, then Phase 2's (20000)
reproduction gate. Those prove the architecture on one coupling before anything
scales to twelve.
