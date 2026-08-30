# S8 source-fixture eligibility audit

## Question

After the unrestricted recursion and node-basis leakage results, the next item
was to acquire an independent physical holdout. The required distinction is
between a published higher-dimensional component system and a valid
one-dimensional Garden construction with no stated higher-dimensional parent.

## Primary-source result

Five local primary sources were audited by page locator and PDF hash:

| Source | Relevant result |
|---|---|
| arXiv:1210.0478 | Classifies the six `GR(4,4)` permutation quartets and treats higher-dimensional equivalence as a separate problem. |
| arXiv:1608.07864 | Identifies `CM`, `VM`, and `TM` as reductions of 4D N=1 multiplets, while calling `VM1`, `VM2`, and `VM3` the other types. |
| arXiv:2012.14015 | Constructs `DO(8)`, or `O`, independently of combinations of N=1 multiplets, without supplying four-dimensional component laws for it. |
| arXiv:2304.09830 | Studies recursive one-dimensional adinkra matrices and identifies `O` with the Rana subgroup. |
| arXiv:2408.09342 | States directly that `VM1`, `VM2`, and `VM3` were derived solely as mathematical Garden-equation solutions, unlike the 0-brane reductions. |

This produces the following eligibility classes:

| Class | Systems | Status |
|---|---|---|
| sourced higher-dimensional positive controls | `CT`, `CV` | exact component fixtures available and passing |
| Garden-positive control without a claimed higher-dimensional parent | `O` | valid one-dimensional control, not a physical holdout |
| mathematical S4 Garden sectors | `VM1`, `VM2`, `VM3` | no stated 4D component parent in the audited corpus |
| printed nonclosing S8 controls | `CC`, `TT`, `TV`, `VV` | printed assignments fail Garden closure |

## Exact gate status

- `CT` and `CV` both pass their sourced four-dimensional component closures,
  gauge residues, and exact worldline reductions.
- The `O` Garden assignment closes exactly.
- The unrestricted recursion and node-basis leakage audits still pass.
- The stated-parent positive-control gate is complete.
- No new independent physical holdout fixture was found.
- A broader physical S8 scan remains unauthorized.

The previous phrase "seven-control higher-dimensional gate" was too broad.
`O` is not missing a fixture for a parent asserted by its source, and the four
printed nonclosing systems are not physical positive controls. The applicable
higher-dimensional positive-control gate currently consists of `CT` and `CV`,
and it is complete.

## What unblocks the next computation

At least one independent target must supply:

1. complete Lorentz representations for all component fields;
2. spatial-derivative linkage coefficients;
3. gauge transformations, field strengths, Bianchi identities, and closure
   residues; and
4. the temporal-gauge field-to-node reduction map.

Without those data, generating spatial transformations from a valise would
fit the target to the candidate and would not constitute a holdout test.

## Reproduction

```sh
cargo run --release -- perm-s8-source-fixture-audit-build
cargo run --release -- perm-s8-source-fixture-audit-verify
cargo test --release permutahedron_s8_source_fixture_audit
```

Artifact:

- `results/permutahedron_s8_source_fixture_audit.json`

## Boundary

The audit does not prove that `O`, `VM1`, `VM2`, or `VM3` can never have a
higher-dimensional realization. It establishes that the reviewed sources do
not provide the independent physical target data required for this test.
