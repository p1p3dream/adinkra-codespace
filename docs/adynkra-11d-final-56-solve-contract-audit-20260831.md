# Audit of the final 56-column four-form solve contract

Date: 2026-08-31

## Verdict

**BLOCKED FOR PHYSICAL SOLVE LAUNCH.** The current artifacts certify the
abstract multiplicities, four exact `(0,2)` streams, and 52-dimensional
three-prime witness rank for the antisymmetrized `(2,1)` diagram span. They do
not yet define one immutable 56-column physical coefficient matrix. In
particular, the selected `(2,1)` maps have witness minors but no complete
column streams or column bindings, and no implemented adapter joins the
physical teleparallel stream to the 56-row schema.

A witness-rank rerun may launch as a diagnostic. A report described as the
final physical solve must not launch until Gates L1 through L8 below close.

## Certified inputs

### Candidate domain

`src/eleven_dimensional_four_form_56_gpu.rs` declares 56 coefficients:

* columns 0 through 51: `(2,1)`, sector multiplicities
  `00001:7, 00011:7, 00101:11, 01001:14, 10001:13`;
* columns 52 through 55: `(0,2)`, sectors
  `00001:1, 01001:1, 10001:2`.

The raw canonical coordinate space is:

| branch | source coordinates | target coordinates | raw rows |
|---|---:|---:|---:|
| `(2,1)` | `496 * 11 * 320 = 1,745,920` | 10,560 | 18,436,915,200 |
| `(0,2)` | `66 * 320 = 21,120` | 10,560 | 223,027,200 |
| total | 1,767,040 | 10,560 | 18,659,942,400 |

These are generator-evaluation coordinates. They are not yet physical
constraint rows and they contain no right-hand side.

### `(2,1)` evidence

`results/adynkra_11d_d21_gpu_witness_ranks.json` reports three-prime ranks
equal to all 52 abstract multiplicities and exact nonzero selected minors.
Its own boundary correctly says that it does not bind final global column
streams or solve the physical coefficient equations.

`results/adynkra_11d_d21_seed_inventory.json` is still an abstract contract:
`passed_cartesian_generator_construction=false`. Its canonical slot order is
Fierz-channel-major, then target sector, then copy. The GPU module assigns
global columns sector-major and `FourForm56ColumnBinding` records only sector
and copy. Consequently the current binding schema cannot distinguish, for
example, a scalar `01001` map from a `Lambda3` or `Lambda4` `01001` map.

The recent exact checks establish the corrected tensor convention for the
diagram evaluator: total four-form antisymmetrization, metric lowering on an
H index contracted with a target output, and no metric on a lower momentum
index routed to a lower target output. They do not replace full stream
publication. The scalar boost canaries are not an exhaustive 55-generator
Lorentz-commutator certificate for all 52 selected maps.

### `(0,2)` evidence

`results/adynkra_11d_d02_00001_source_generator.json` and
`results/adynkra_11d_d02_remaining_seed_inventory.json` provide the four
exact streams. The latter computes the four-column spacetime Bianchi matrix:
rank two at all three primes, exact rank two, and exact kernel dimension two
with replayed rational kernel vectors. Thus only a two-dimensional
combination of the four `(0,2)` generators is closed. The four raw columns
must not be passed directly to a physical solve without either appending the
Bianchi rows or changing to a certified Bianchi-kernel basis.

### Physical target evidence

`corrected_full_chain_streams` in
`src/eleven_dimensional_corrected_full_chain_oracle.rs` emits the corrected
Eq. (40) candidate and teleparallel `D G4` target using
`FullChainRowKey(output_coordinate, exterior_spinor_mask,
momentum_exponents)`. It checks `DDelta=D(Delta)` and Eq. (25) curl equals Eq.
(29). It does not emit `CanonicalRow(branch, source_coordinate,
target_coordinate)` and it is callable for only one H-hat ordinal at a time.

No current function performs the exact all-320 conversion:

```text
FullChainRowKey plus H-hat ordinal
    -> exterior-spinor pair or symmetric-momentum pair
    -> (2,1) or (0,2) source_coordinate
    -> CanonicalRow
```

That join is load-bearing. It must pin the right-C convention, ordered PBW
mask convention, momentum monomial order, H-hat basis, 496 exterior-spinor
pair order, 66 symmetric-pair order, and the diagonal/off-diagonal symmetric
pair normalization.

## Correct solve formulation

Let `M` be the exact 18,659,942,400 by 56 sparse candidate map and let `t` be
the corrected physical teleparallel stream in exactly the same row basis.
Before an authoritative component normalization is available, solve

```text
[ M | -t ] (c_0,...,c_55,s)^T = 0.
```

Interpret the result fail-closed:

* nullity zero: the physical target is not in the 56-map span;
* nullity one with `s != 0`: a unique map exists up to scale;
* any kernel vector with `s = 0`: an internal ambiguity of the ansatz;
* nullity greater than one: the physical adapter is not unique;
* a unique ray is not normalized until an authoritative equation fixes `s`.

If the physical target normalization is already authoritative, set `s=1`
and require `rank(M)=rank([M|t])` at every prime. In either formulation, lift
the modular solution to `Q(i)` and replay every original exact sparse row.
Three-prime rank agreement alone is not an exact survivor certificate.

## Required row families

The following families must be appended to the solve or independently
certified on every candidate basis column and on the physical right-hand
side. If full all-row equality to a certified target makes a family
mathematically redundant, retain it as an independent integrity gate and
mutation oracle.

1. **Physical stream equality.** All `(2,1)` and `(0,2)` PBW rows of `M c-s t`.
   A support-only comparison is insufficient.
2. **Spacetime four-form Bianchi.** `p wedge D G4=0` for both branches. The
   `(0,2)` four-column matrix exists. The corresponding `(2,1)` Bianchi image
   and its stream hash do not.
3. **PBW and descendant integrability.** Reapply the ordered-superderivative
   algebra, including the symmetric-D anticommutator momentum term, to the
   selected 56 streams. Require the branch join, `DDelta=D(Delta)`, and Eq.
   (25)/Eq. (29) identities in the common convention. Independent branchwise
   Hom bases do not by themselves certify that they are descendants of one
   physical superfield map.
4. **Gamma-trace/source-section consistency.** Verify all rows on the full
   352 frame-spinor lift or prove that every section change killed by
   `P_320` maps to zero. A calculation on the chosen 320-dimensional section
   alone does not prove section independence.
5. **Source gauge and local Lorentz descent.** For each declared source
   redundancy, compose the candidate with the exact source-gauge map. Because
   `G4` is a field strength, the result must vanish, or be routed through an
   explicitly constructed target-gauge/reducibility map. Global Lorentz
   equivariance is not local Lorentz gauge invariance.
6. **Three-form target reducibility.** If the solve passes through `A3`, bind
   the `A3 -> G4` exterior derivative and verify independence of the
   `A3 -> A3+p wedge Lambda2` representative. If it works directly in `G4`,
   record that target gauge is already quotiented and verify the same result
   by an exact lift canary.
7. **Supersymmetry descendant matching.** Match the standard `D_alpha G4`
   constraint to the direct gravitino curl/torsion using the corrected right-C
   dictionary. Then match the next descendant to the frame/Riemann branch.
   This is the relative physical normalization gate.
8. **Euler and Noether compositions.** State explicitly whether the solve is
   off-shell identification or an on-shell restriction. If Euler rows are
   imposed, require the exact Euler-to-Noether zero composition. Do not use
   equations of motion silently to manufacture a unique map.

## Launch gates

### L1: immutable 56-column inventory

Publish all 52 selected `(2,1)` exact streams, not only pivot minors. Version
an explicit permutation from abstract Fierz-channel-major seed slots to global
sector-major columns, or change the global order. Extend each D21 column
binding with Fierz channel, selected raw diagram/orbit semantic identity,
target sector, copy, projector hash, exact entry count, and exact stream hash.
The ordered 56-binding digest must reject missing, duplicated, or permuted
columns. Close the inventory's Cartesian-construction gate with either all 55
Lorentz commutators on the complete source basis or a machine-checked proof
that every selected map is composed only of the already certified equivariant
tensor primitives. Retain the old H-output-delta convention as a rejecting
mutation.

### L2: physical-row adapter

Convert all 320 corrected teleparallel streams to `CanonicalRow`, emit a
deterministic exact target digest, and reject every row outside the declared
`(2,1)` and `(0,2)` branches. Publish per-branch row counts and first/last row
witnesses. A nonzero outside branch is a bidegree-exhaustion blocker, not a
row to discard.

### L3: denominator ledger

Compute one declared Gaussian-integer lattice, audit every generator,
projector, physical-target, and normalization denominator against all three
pinned primes, and bind the ordered denominator digest into the manifest.

### L4: closure and integrability

Publish Bianchi, PBW branch-join, gamma-trace/section, and descendant
constraint hashes. For a prefiltered basis, publish the exact change-of-basis
matrix and reconstruct the unfiltered rows.

### L5: source and target descent

Publish exact compositions for every applicable source redundancy and the
three-form target reducibility complex. No section dependence may remain
implicit.

### L6: three-prime solve

Stream canonical rows in deterministic blocks. At every prime report rank,
nullity, pivot columns, a nonzero actual-row minor, RHS consistency, and
whether every kernel vector has `s=0` or some has `s!=0`. Never cap RREF at an
expected rank without an expected-plus-one sentinel.

### L7: characteristic-zero reconstruction

Reconstruct the canonical coefficient vector over `Q(i)`, normalize it only
from a pinned physical equation, and replay all original rows plus every
constraint family exactly. Record the first mismatch on failure.

### L8: completeness and publication

Bind a nonnull bidegree-exhaustion certificate for any survivor claim. Use
atomic checkpoint/status publication, deterministic resume, fail-fast shared
cancellation, and report-last final publication. Mutations must reject at
least: a swapped D21 column permutation, wrong off-diagonal momentum-pair
normalization, omitted PBW branch link, omitted D21 Bianchi family, stale
physical-target hash, wrong right-C adapter, source-section mutation, and one
corrupted reconstructed coefficient.

## Smallest safe implementation order

1. Freeze the corrected D21 selected-map permutation and emit all 52 streams.
2. Build the all-320 `FullChainRowKey -> CanonicalRow` physical target adapter.
3. Run a single-column and then one-row-per-family CPU exact join canary.
4. Build D21 Bianchi and PBW/descendant constraint emitters; reuse the existing
   D02 Bianchi emitter.
5. Add source-section and source-gauge descent rows.
6. Launch the 57-column homogeneous three-prime streamed solve on the GPU.
7. Reconstruct and replay the canonical `Q(i)` ray on CPU.
8. Fix scale only from the gravitino/graviton or pinned superspace equation,
   then publish the report atomically.

The decisive immediate blockers are L1 and L2. Until both close, GPU rank is
only a generator-span diagnostic, not the physical 56-coefficient solve.
