# Maxwell gauge-enhancement scan of the published S4 signings

## Result

The validated Maxwell worldline search has been applied to all 96 fiducial
signed quartets in Appendix B of arXiv:1701.00304.

Every input satisfies the Garden algebra. For each input, the search examines
all 147,456 signed boson and fermion frame pairs and evaluates the complete
Maxwell gauge-enhancement condition from arXiv:0907.3605.

| Quantity | Count |
|---|---:|
| published fiducial signings | 96 |
| Garden-closing inputs | 96 |
| signed frame pairs per input | 147,456 |
| total signed frame pairs | 14,155,776 |
| gauge-enhancing signings | 48 |
| rejected signings | 48 |

The scan gives one clean split:

```text
Maxwell gauge-enhancement passer if and only if chi0 = -1
```

All 48 signings with `chi0 = -1` pass. All 48 signings with `chi0 = +1`
fail. Each of the six unsigned quartets contains eight signings of each class,
so each unsigned quartet contributes eight passers and eight failures.

## Interpretation

This reproduces the expected cis/trans separation in the complete published
fiducial library. It does not supply a new selector beyond `chi0` in this
four-color dataset.

The labels `CM`, `TM`, `VM`, `VM1`, `VM2`, and `VM3` name unsigned permutation
quartets here. Each unsigned quartet admits both `chi0` classes among its 16
published signings. Therefore, the result does not mean that every unsigned
quartet has a Maxwell parent. The sign data remain essential.

None of the published signings passes in its stored component frame. The
source-frame normalization is required. After normalization, every input has
384 candidates, and each `chi0 = -1` input has eight passing signed frames.

## Independent cross-check

The JavaScript implementation separately reconstructs the gamma matrices,
phantom linkages, spatial linkages, Omega tensors, Bianchi reshuffling, and all
14,155,776 signed frame pairs. It reproduces the 48-to-48 split and its exact
correlation with `chi0`.

## Reproduction

```sh
cargo run --release -- maxwell-s4-atlas-build
cargo run --release -- maxwell-s4-atlas-verify
cargo test --release maxwell_s4_atlas_scan
node scripts/test_maxwell_worldline_search.mjs
```

Artifact:

- `results/maxwell_s4_atlas_scan.json`
- SHA256: `6336526904a9b74b82a0e8887880c0647d93a8b39fa293ba3b41d6f497b97441`

## Consequence

At four colors, the full Maxwell phantom and Bianchi calculation agrees exactly
with the simpler `chi0` classification on this finite library. Applying the
four-color gate mechanically to eight-color matrices would therefore not add
information.

The next useful calculation is to retain the two embedded four-color
subalgebras of each eight-color construction and record their ordered pair of
gauge-enhancement classes. That can test whether the recursive pair structure
retains information that the complete eight-color Garden class loses.

## Boundary

This exhausts the 96 published fiducial signings, not all rankings or all BC4
transformations. The equivalence with `chi0` is established only for this
library. It is not a proof that every four-color Adinkra with `chi0 = -1` has a
specific higher-dimensional parent.
