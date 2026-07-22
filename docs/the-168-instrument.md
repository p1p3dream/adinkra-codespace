# The 168: an S8 permutahedron instrument

`visualizer/the_168.html` is a browser instrument for the S8
permutahedron and the 168 left-right coincident R8 cosets inside it.
It renders all 40,320 vertices and 141,120 edges as a genuinely
7 dimensional object, rotated live in SO(7) on the GPU, shown as an
exact 3D orthographic slice. It is built to be shown to a physicist
and to never claim more than the checked-in data certifies.

## How to run

From the repository root:

```
python3 -m http.server 8168
```

then open

```
http://localhost:8168/visualizer/the_168.html
```

Any static file server works. The page must be served over http
(not opened as file://) because it fetches the JSON datasets with
relative paths. Three.js r164 is vendored at
`visualizer/vendor/three.module.js`, so no network is needed at demo
time.

To run the test suite:

```
node scripts/test_the_168.mjs
```

It prints PASS or FAIL per check (SO(7) orthogonality and determinant,
exact sqrt(2) edge lengths in the 7D embedding, adjacency
preservation, the 168 igniter selection, the Ledger numbers against
the JSON) and exits nonzero on any failure.

## What it shows

- The permutahedron of S8 embedded in the sum-zero hyperplane of R^8
  via an exact orthonormal (Helmert) basis: 40,320 points in 7D,
  computed once at load in `visualizer/rotor7.mjs`. Each dataset edge
  has geometric length sqrt(2); the object on screen is the honest
  polytope skeleton, not a force layout.
- A live SO(7) rotation composed from angles in the 21 coordinate
  planes, applied in the vertex shader together with the 7D to 3D
  orthographic projection (`visualizer/the_168_render.mjs`). Dragging
  steers two of the 21 plane angles; nothing ever leaves SO(7).
- A nine-beat guided narrative, "From 24 to 168"
  (`visualizer/the_168_narrative.mjs`): S4 warm-up (the truncated
  octahedron, genuinely 3D), pair distance measurement, the ascent to
  S8, the crystallization of the 5,040 right cosets, the left/right
  moire with the coincidence foreshadow, a named published octet, the
  closure verdict panel, and the gold ignition of the 168 with the
  identification 168 = 1344 / 8 = |GL(3,2)|. The demo ends on an open
  question, not a claim.
- GPU color-ID picking: clicking a vertex reads its rank from an
  offscreen ID render, then shows its address, stable id, and coset
  memberships. Clicking two vertices shows their minimal weak-Bruhat
  distance, labeled in-UI as exactly that and explicitly not a
  holoraumy gadget (quoting `metadata.correlator_definition`).

## The honesty framing

- A permanent badge reads: "Genuine 7D object. You are seeing an
  exact 3D orthographic slice." That is literally what the shader
  does: rotate in 7D, keep three coordinates.
- The Ledger (always one click away) has two columns of equal weight.
  Reproduced: vertex and edge completeness as verified by the in-page
  check, the 5,040 cosets, rank 45 / nullity 19 / 2^19 signings per
  coset, the sixteen honest dashing classes (2^19 raw = 2^15
  node-sign gauge x 2^4; 524,288 is never presented as distinct
  physics), the 168 and the GL(3,2) identification, and the seven
  named octets with their published statuses. Not claimed: the
  SUSY-weight-space reading is a research proposal, not a theorem;
  the scan's own `boundary` and `conclusion` strings are quoted
  verbatim; the pair distance is not a holoraumy gadget; no new
  off-shell result.
- Provenance is displayed: the pinned arXiv ids and PDF sha256 hashes
  from `metadata.source`, plus a note that the page renders the
  repository JSON directly.
- On load the page runs `runSelfChecks()` and `validateDataset()` on
  both atlases and gates on `garden.passed === true`,
  `cosets_scanned === signable_cosets === 5040`, and exactly 168
  abnormal slices. Any failure raises a loud red banner and halts;
  it never silently proceeds.
- Color discipline: gold appears nowhere in the interface except on
  the 168. Green and amber appear only in the closure verdict.

## Embedding convention (why the picture is exact)

The dataset's edges are right multiplication by adjacent
transpositions (adjacent-position swaps). Embedding each permutation
by the tuple of its inverse turns every such edge into a swap of two
consecutive values in the embedded coordinates, which is a geometric
permutohedron edge of Euclidean length sqrt(2). This is a relabeling
of the same abstract Cayley graph, not a different object, and the
test suite verifies it edge by edge.

## Files

- `visualizer/the_168.html`: the instrument shell, checks, UI.
- `visualizer/rotor7.mjs`: pure 7D embedding and SO(7) rotor math.
- `visualizer/the_168_render.mjs`: Three.js scene, shaders, picking,
  the two set-piece animations (crystallize, ignite).
- `visualizer/the_168_narrative.mjs`: the nine beats and the Ledger
  content, all data-driven.
- `visualizer/vendor/three.module.js`: Three.js r164, vendored.
- `scripts/test_the_168.mjs`: the test suite.

Existing files (`visualizer/permutahedron_atlas.html`, the 2D atlas)
are untouched; `visualizer/permutahedron_core.mjs` is reused as-is.

## Phase 1 scope, deferred to Phase 2-3

Deferred deliberately:

- Live WASM recompute-and-byte-diff verification of the datasets in
  the browser (Phase 1 verifies structure in JS and gates on the
  scan's own pass flag).
- The full 21-plane "7-Rosette" rotor control UI (Phase 1 exposes
  drag on two planes plus the idle drift).
- The full six-stage physics unfold drawer.
- The full thirteen-invariant "why 168?" comparison lab.
- Sound design (the UI is silent in Phase 1).
