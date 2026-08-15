# Unrestricted S8 recursion and normalizer-orbit leakage

## Unrestricted Boolean-mask census

The aligned signed S8 recursion was expanded from the eight published cyclic
weight-four masks to all 256 masks. The six same-source ordered pairs were also
added as controls.

| Quantity | Count |
|---|---:|
| source labels | 6 |
| ordered pairs | 36 |
| relative color orders per pair | 24 |
| masks per alignment | 256 |
| exact candidates checked | 221,184 |
| Garden closers | 64 |
| distinct closing supports | 32 |
| closing realizations per support | 2 |
| same-source closers | 0 |
| noncyclic closers | 40 |
| additional supports beyond the cyclic scan | 20 |

All 64 closers come from ordered distinct pairs. There are no mixed-source
closers: 32 realizations come from named-parent pairs and 32 from
unstated-parent pairs. Every closing support belongs to family 0.

The original restricted calculation is recovered exactly: 24 cyclic-mask
closers on 12 supports.

## Fixed-basis normalizer correlation

The unrestricted 32-support set occupies five normalizer orbits:

| orbit | supports | source category |
|---:|---:|---|
| 1 | 4 | unstated-parent pair |
| 4 | 4 | unstated-parent pair |
| 5 | 8 | unstated-parent pair |
| 7 | 8 | named-parent pair |
| 17 | 8 | named-parent pair |

Thus the source-category correlation survives unrestricted masks and gains
orbit 4 on the unstated-parent side.

## Node-basis leakage audit

The fixed-basis correlation is not intrinsic to the unlabeled valise. Starting
from one exact closing support `H*g`, a common relabeling of one node level
postcomposes every color permutation by the same `b` in S8:

```text
H*g -> H*(g*b)
```

The exhaustive calculation applies all 40,320 relabelings and finds:

- all 40,320 transformed supports map back into family 0;
- all 5,040 family-0 supports are reached;
- every target support is reached exactly eight times;
- all 20 normalizer orbits are reached; and
- normalizer-orbit ID is therefore not invariant under common node relabeling.

## Consequence

Normalizer orbit is an exact coordinate of the published component basis and
describes the recursion construction cleanly. It is not an invariant of the
unlabeled worldline representation. The orbit-to-source-category correlation
cannot serve as an intrinsic physical parentage selector unless a component
basis is fixed independently by higher-dimensional data.

The unrestricted scan strengthens the construction-level correlation, while
the leakage audit closes it as a basis-independent selector.

## Reproduction

```sh
cargo run --release -- perm-s8-unrestricted-recursion-build
cargo run --release -- perm-s8-orbit-leakage-build

cargo test --release permutahedron_s8_unrestricted_recursion
cargo test --release permutahedron_s8_orbit_leakage
```

Artifacts:

- `results/permutahedron_s8_unrestricted_recursion.json`
- `results/permutahedron_s8_orbit_leakage.json`

## Boundary

The unrestricted census is exhaustive within the aligned block-recursion
ansatz. It is not an enumeration of every `2^64` sign assignment or every S8
construction. The leakage audit does not make a claim about a component basis
that is supplied independently by a higher-dimensional field theory. It shows
that the orbit label cannot be recovered intrinsically from the unlabeled
valise alone.
