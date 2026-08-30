# S8 recursion, Maxwell, and hypergraph bridge

## Result

The closing candidates of the signed S8 recursion have been mapped into the
complete 151,200-edge constraint hypergraph and joined to the exact Maxwell
classification of their two retained four-color blocks.

| Quantity | Result |
|---|---:|
| aligned recursion candidates | 5,760 |
| S8 Garden closers | 24 |
| distinct unsigned supports | 12 |
| closing recursive signings per selected support | 2 |
| discovered hypergraph families occupied | 1 |
| family histogram | family 0: 24 |
| normalizer orbits occupied | 4 of 20 |
| embedded S4 blocks | 48 |
| embedded S4 blocks closing | 48 |

Every closer pairs one `chi0=+1` block with one `chi0=-1` block. The complete
four-color Maxwell phantom and Bianchi gate selects exactly the `chi0=-1`
block. The 24 closers split into two ordered signatures of 12 candidates each:

```text
(-1, +1) -> (pass, fail)
(+1, -1) -> (fail, pass)
```

Both signatures mix source pairs with named four-dimensional parents and
source pairs without stated parents. The published CT and CV anchors occupy
different unsigned supports but both have `(fail, pass)`.

The normalizer-conjugacy refinement gives a sharper finite-library split:

| normalizer orbit | selected supports | recursive closers | source category |
|---:|---:|---:|---|
| 1 | 4 | 8 | unstated parent |
| 5 | 2 | 4 | unstated parent |
| 7 | 4 | 8 | named parent, including CV |
| 17 | 2 | 4 | named parent, including CT |

Every occupied normalizer orbit contains both ordered Maxwell signatures. The
normalizer orbit therefore separates the named-parent and unstated-parent
source categories in this finite recursion library, while the embedded
Maxwell signature does not. It also distinguishes the CT and CV supports:
`CT` is in orbit 17 and `CV` is in orbit 7.

## Interpretation

The recursion makes a genuine finite selection inside family 0: 12 supports,
each carrying two closing recursive signings. The normalizer orbit is a
nontrivial stratifier of that selected set. The embedded Maxwell label is
exactly the four-color `chi0` label and does not distinguish CT from CV.

The subsequent unrestricted-mask scan strengthens the fixed-basis correlation:
64 closers on 32 supports remain category-pure across orbits 1, 4, 5, 7, and
17. However, the node-basis leakage audit closes it as an intrinsic selector.
A common relabeling of one node level carries any selected support through all
5,040 family-0 supports and all 20 normalizer orbits. Orbit ID is therefore not
an invariant of the unlabeled valise.

The combined result closes three proposed intrinsic selectors:

1. membership in the standard R8 family;
2. the ordered Maxwell classes of the retained S4 blocks; and
3. normalizer-conjugacy orbit without an independently fixed component basis.

Normalizer orbit remains a useful source-basis coordinate. See
`docs/permutahedron-s8-unrestricted-recursion-and-orbit-leakage.md` for the
complete stress test.

## Reproduction

```sh
cargo run --release -- maxwell-phantom-build
cargo run --release -- maxwell-worldline-search-build
cargo run --release -- maxwell-s4-atlas-build
cargo run --release -- maxwell-s8-subalgebra-build
cargo run --release -- perm-hypergraph-recursion-maxwell-build

cargo test --release maxwell_phantom
cargo test --release maxwell_worldline_search
cargo test --release maxwell_s8_subalgebra_scan
cargo test --release permutahedron_hypergraph_recursion_maxwell_bridge

node scripts/test_maxwell_phantom.mjs
node scripts/test_maxwell_worldline_search.mjs
```

Primary artifacts:

- `results/maxwell_phantom.json`
- `results/maxwell_worldline_search.json`
- `results/maxwell_s4_atlas_scan.json`
- `results/maxwell_s8_subalgebra_scan.json`
- `results/permutahedron_hypergraph_recursion_maxwell_bridge.json`

## Boundary

The Maxwell gate applies to the two four-color blocks retained by the first
four colors of the recursive construction. The complete eight-color
representation is irreducible. A block-level Maxwell passer does not establish
an N=8 higher-dimensional parent. A full test still requires an independently
specified Lorentz representation, spatial linkage, gauge complex, phantom
inventory, and reduction map.
