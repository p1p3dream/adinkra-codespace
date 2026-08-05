# S4 disassembly: phased build plan

Written 2026-08-05 from the 2026-08-04 spec call
(`plan-sources/gates-20260804-permutahedron-spec-call.txt`, line numbers cited
as L*) and the 2026-07-30 styling call
(`plan-sources/gates-20260730-styling-call-lines-1-77.txt`, cited as SL*).
Supersedes the previous version of this file. The one open contradiction in
that version (where the animation journey enters each quartet) is resolved
below and the wrong branch is deleted.

Gates said on 2026-07-30: "There's no rush. This is already super fast from my
perspective" (SL69-71). The overriding constraint is correctness. He already
caught one ordering error in a delivered build. Nothing in this plan is
scheduled against a clock.

---

## Ground rules for every phase

1. **Phases are sequential.** Each phase ends in an exit gate. The next phase
   does not start until the gate passes.
2. **Exit gates must be able to fail.** The current `selfCheck()`
   (`visualizer/permutahedron_s4_disassembly.template.html:253`) has ten
   conditions; eight are bare reads of booleans baked in by the build script
   (`visualizer/build_permutahedron_s4_disassembly.mjs:232-245`). A baked
   boolean read at render time cannot fail, and that is how four review passes
   cleared two false captions. Every new gate assertion must be recomputed from
   the rendered data or geometry at check time, and every new assertion must be
   proven falsifiable once: break the input deliberately, watch the gate fail,
   restore the input, watch it pass.
3. **Verify quartets 4 and 6. Never spot-check quartet 1.** Quartet
   `1234, 2143, 3412, 4321` is identical under ascending weight, final-digit
   ordering, and hopper ordering. It is the Vierergruppe, so it is what
   everyone naturally checks, and three separate errors survived review because
   the one case checked is the one case that cannot fail. Any phase touching
   the six quartets must verify quartets 4 (`1342, 2431, 3124, 4213`, base
   member `3124` in 3rd position) and 6 (`1432, 2341, 3214, 4123`, base member
   `3214` in 3rd position) specifically, and must not use quartet 1 as
   evidence of anything.
4. **Each phase is independently deliverable.** If work stops after any phase,
   what exists is correct and showable.
5. **Guard rails carried over from the superseded plan, still correct:**
   do not restore the hopper ordering (he asked for ascending weight; hopper
   ordering gives uniform 2,6,2 legs, his gives six distinct patterns), and do
   not delete the per-leg distance labels (under his ordering the leg patterns
   differ per quartet, so the labels carry information).

## Fixed reference data

Ordering rule: read each permutation as a four-digit number, sort ascending
(L183-187). Authority is Howard talk page 61 (L199-201). Path rule: shortest
path, never backtrack, "the self-avoiding random walk" (L147-161).

The six quartets exactly as HowardTLK.v2.pdf p. 61 prints them, P[1] to P[6],
with consecutive leg link-distances and base-hexagon membership:

| # | page 61 | quartet | legs | base member, position | multiplet |
|---|---|---|---|---|---|
| 1 | P[1] | 1423, 2314, 3241, 4132 | 4, 2, 4 | 2314, 2nd | CM |
| 2 | P[2] | 1342, 2431, 3124, 4213 | 6, 4, 6 | 3124, 3rd | TM |
| 3 | P[3] | 1324, 2413, 3142, 4231 | 4, 6, 4 | 1324, 1st | VM |
| 4 | P[4] | 1432, 2341, 3214, 4123 | 6, 2, 6 | 3214, 3rd | VM1 |
| 5 | P[5] | 1243, 2134, 3421, 4312 | 2, 4, 2 | 2134, 2nd | VM2 |
| 6 | P[6] | 1234, 2143, 3412, 4321 | 2, 6, 2 | 1234, 1st | VM3 |

**Two ordering rules, and conflating them is the trap.** Within a quartet the
members ascend as four-digit numbers. The quartets themselves run P[1] to P[6],
which is *not* a smallest-member sort: their first members run 1423, 1342, 1324,
1432, 1243, 1234. An earlier revision of this plan and of the builder sorted the
quartets by smallest member, which produced VM3, VM2, VM, TM, CM, VM1 and
disagreed with the page Gates named as authoritative. Both the builder and the
verifier now assert that the first members are *not* ascending, so that mistake
cannot come back silently.

The multiplet column matches arXiv:1210.0478 Eq. (5.2a-f) and the colour swatches
printed on p. 61 (CM green, TM purple, VM red, VM1 blue, VM2 orange, VM3 teal).

Base face: the six permutations ending in 4. Cyclic order:
`1234, 1324, 3124, 3214, 2314, 2134`.

An earlier draft of this plan claimed the cyclic order was
`1234, 1324, 2314, 3214, 3124, 2134`, and called the stored order not cyclic on
the grounds that `1324-3124` and `2314-2134` were chords of length 2.828 rather
than edges. That is wrong. Both pairs are genuine permutahedron edges, every
consecutive pair in the stored order is adjacent, and the cycle closes. The
order already in the build script was correct, and applying the claimed fix
would have introduced a self-intersecting hexagon into working code.

The error was caught on the first run of `baseFaceIsCyclic`, which walks the
listed order against the atlas edge list. Do not re-derive this by inspecting
coordinates; assert adjacency against the edges.

**Resolved, supersedes the old section 2 of this file:** the strand listing is
ascending weight AND the preset animation enters each quartet at position 1,
traversing 1 to 2 to 3 to 4 cumulatively, retracing and extending exactly as he
demoed (L11-15, L43-55). The earlier idea that the journey enters at the
base-face node is wrong and is deleted; it would make quartets 2, 4, 5, and 6
fill out of order. Base-face membership is a highlight, not an ordering rule.
One question remains for him (see Open questions): confirm entry at position 1
versus the hexagon node.

Jul 30 styling contract, previously undocumented, now pinned here so it cannot
be lost: link segments thickened to roughly the weight shown on Howard talk
page 64 (SL29; the page reference is SL21, and it is page 64, not 61), and node
names rendered inside the node domains in white letters (SL7, SL31). He then
said "Then that's all I need" (SL33). The current bubble renderer draws dark
text on white spheres (`permutahedron_s4_disassembly.template.html:168`), which
does not meet this contract.

---

## Phase 1: Preserve

Everything else is blocked on this. A single `git clean -fd` currently
destroys every artifact Gates has seen.

**Precondition:** none.

**Work:** put the untracked visualizer work under version control. Verified
untracked as of 2026-08-05 (`git status -s visualizer/`):

- `visualizer/permutahedron_s4_disassembly.template.html`
- `visualizer/permutahedron_s4_explorer.template.html`
- `visualizer/permutahedron_s8_explorer.template.html`
- `visualizer/permutahedron_s8_webgl_core.mjs`
- `visualizer/build_permutahedron_s4_disassembly.mjs`
- `visualizer/build_permutahedron_s4_explorer.mjs`
- `visualizer/build_permutahedron_s8_explorer.mjs`
- `visualizer/permutahedron_s4_disassembly.html`
- `visualizer/permutahedron_s4_explorer.html`
- `visualizer/permutahedron_s8_explorer.html`

Track the generated `.html` outputs too, not just the templates: they are the
exact artifacts Gates has interacted with, and a rebuild is not guaranteed
byte-identical. Record the SHA-256 of each generated file in the commit
message so the delivered state is pinned. The input JSONs
(`data/permutahedron_s4_atlas.json`, `data/permutahedron_s4_supersymmetry.json`)
are already tracked. The untracked `viz/` directory (python 3D plot scripts
and PNGs) is separate work; review and either track or gitignore it in the
same pass so `git status` comes out clean.

**Exit gate:**
- `git ls-files visualizer/` lists all ten files above.
- `git clean -fdn` (dry run) reports nothing under `visualizer/`.
- `git status -s` shows no untracked files in `visualizer/`.

---

## Phase 2: Correct what is currently shipped wrong

The shipped build orders strands by hopper words, not ascending weight, and
carries two false captions and a clipping bug that hid the 4th element of
every quartet during the demo.

**Precondition:** Phase 1 gate passed.

**Work:**
1. **Reorder every strand to ascending weight** per the fixed reference table.
   The current chain seeds and hopper walk
   (`build_permutahedron_s4_disassembly.mjs:19,23-28`) generate H1-H4 hopper
   order with uniform 0,2,4,6 root distances; replace with the ascending
   listing and regenerate all leg distances by BFS on the atlas graph. Do not
   carry over any number from the previous ordering.
2. **Replace the uniform-legs validation.** The build currently requires every
   quartet to have consecutive distances `2,6,2`
   (`build_permutahedron_s4_disassembly.mjs:211,220`). That is the hopper
   ordering's signature and is wrong under ascending weight. Replace with the
   per-quartet expected leg table above, keyed by first element.
3. **Leave the base hexagon cyclic order alone.** It was already correct. Add
   `baseFaceIsCyclic`, which walks the listed order against the atlas edge list
   and fails if any consecutive pair, including the wrap from last to first, is
   not an edge. See the correction under the reference table above.
4. **Delete the two stale strings:** the page title "S4 Six-Quartet Ascending
   Separation" (`permutahedron_s4_disassembly.template.html:6`) and the
   subtitle "chains ascending P1-P6 and nodes ascending H1-H4"
   (`permutahedron_s4_disassembly.template.html:72`). Both are false as
   printed. Regenerate captions from computed data rather than hand-writing
   them, so a caption cannot again disagree with the geometry. The export
   header strings (`template:179,181`) and the H1/hopper vocabulary
   throughout the panel text (`template:85,90,115,126-130`) must be reworked
   in the same pass; they all describe the hopper ordering.
5. **Fix the clipping.** `overflow:hidden` on `html,body` (`template:15`) with
   a 1140px minimum grid (`template:26`: 280 + 500 + 360) meant Gates never
   saw the 4th element of any quartet, or any disclaimer, during the demo.
   The layout must degrade by scrolling or reflow, never by silent cropping.
6. **Replace `selfCheck()`** (`template:253`) with assertions recomputed at
   render time from the delivered data: ascending order recomputed by reading
   each strand's four labels as integers, leg distances recomputed by BFS,
   hexagon simplicity recomputed from projected coordinates (consecutive
   pairs adjacent, no segment crossings).
7. **Add the VM2 erratum disclosure** as a visible, non-clippable note (see
   the erratum section below for exact content).

**Exit gate:**
- Rendered-geometry assertions, all recomputed at check time, none read from
  baked booleans: (a) each strand's labels read as integers are strictly
  increasing; (b) BFS leg distances match the reference table for all six
  quartets; (c) the base hexagon's consecutive pairs are graph-adjacent and
  its projected outline has no self-intersection.
- Falsifiability proof: swap members 3 and 4 of quartet 6 in the input, gate
  fails; restore, gate passes. Repeat with one leg-distance entry and one
  hexagon vertex.
- Quartets 4 and 6 verified by hand against the reference table on the
  rendered page. Quartet 1 not used as evidence.
- At the window size used on the 08-04 call, all six strands show four nodes,
  and the erratum note is visible without scrolling tricks.
- The exported PNG includes the strand panel (`exportPng`, `template:243`,
  currently renders only the canvas scene) and is regenerated as proof.
- Page renders non-blank; the self-check throws before render, so a blank
  page is the failure mode to check for explicitly.

---

## Phase 3: Deliver his literal asks

Everything he asked for on 08-04, plus the Jul 30 styling contract.

**Precondition:** Phase 2 gate passed. Ordering is correct before any
animation work; animating the wrong order would burn a second review.

**Work:**
1. **Horizontal strand panel** matching the demo screenshot layout.
2. **Per-strand pop-out on click** (L211-215): six clicks, one per strand,
   nothing auto-advances. He narrates over it; these are lecture materials.
   Remove or demote the current auto-play on load (`template:255`).
3. **Sequential link glow along each journey** (L17-33): the preset enters at
   position 1 and retraces cumulatively (L11-15, L43-55); each link glows as
   the walk reaches it, the animation rests between members (L23), then
   restarts from the entry node and extends.
4. **Blank white ball left at each vacated position** (L219).

   This only reads correctly if the permutahedron edges are drawn on the fixed
   lattice rather than between current node positions. Drawn the other way, an
   extracted strand drags its edges with it and the solid tears apart instead of
   showing a hole. The strands stack to the right and the solid slides left as
   they come out, so the vacated lattice stays legible.
5. **Hexagon at the floor** (L177), using the corrected cyclic order.
6. **Jul 30 styling contract:** thick links per Howard page 64 (SL21, SL29);
   node names inside the node domains in white letters (SL7, SL31).

**Exit gate:**
- Scripted click-through: six clicks produce six pop-outs in listed order;
  after pop k, exactly 4k white balls occupy vacated positions (counted from
  render state, not from a flag).
- Glow order assertion: the glow sequence for each strand equals the
  recomputed shortest-path route for that strand's legs, checked on quartets
  4 and 6 member by member. Quartet 1 is not evidence.
- The glow paths for quartets 4 and 6 never revisit a node (backtrack check
  recomputed from the route arrays).

  **Correction found while building this.** Choosing each leg independently
  satisfies the per-leg condition but produces a concatenated walk that revisits
  a node in quartets 1 (VM3, revisits 4312) and 6 (VM1, revisits 3241). Gates
  applies the self-avoiding condition to the whole traversal, not to each leg,
  so the legs are now chosen jointly: the first combination of per-leg geodesics
  whose concatenation visits no node twice. All six quartets admit one. The
  emitted `journey_ranks` is that walk, and `member_stops` marks where the four
  members sit along it.
- Styling: rendered label pixels inside node fill are white; link stroke
  width matches the Howard page 64 sample by side-by-side comparison. The
  Howard PDF was permission-blocked from this machine during planning; open
  `~/Documents/HowardTLK.v2.pdf` page 64 manually for the comparison and note
  the measured width in the commit.
- Falsifiability proof: reverse one strand's route array, glow-order gate
  fails; restore, passes.

---

## Phase 4: Free selection, the generalization

Brandon's design, beyond what was asked: turn the presets into a teaching
instrument that also scales past S4.

**Precondition:** Phase 3 gate passed. Presets and free selection must
coexist; the presets are the deliverable Gates depends on, so they land first.

**Work:**
1. Free selection of nodes on the assembled permutahedron.
2. When both endpoints of an edge are selected, that edge highlights.
3. A running link count for the selection order, with a backtrack flag: the
   tool teaches the shortest-path, no-backtracking rule (L147-161) without
   enforcing it.
4. An "extract path" action that sends the current selection to the chain
   view as a new strand alongside the six presets.

This design generalizes to the 40,320-node hex, which Gates named as the next
target (L263: "Our next big target will be the omni-truncated seven simplex...
the hex"). Hardcoded quartets do not generalize; selection plus recomputed
distance does, and the S8 substrate already exists
(`visualizer/permutahedron_s8_explorer.template.html`,
`data/permutahedron_s8_atlas.json`).

**Exit gate:**
- Select the two endpoints of a known edge from quartet 6's route (for
  example `3214` and `3124`): the edge highlights. Select two non-adjacent
  nodes: no edge highlights.
- Select `1342` then `4213` (quartet 4 endpoints): link count equals the BFS
  distance recomputed at check time.
- Construct a deliberate backtrack (`1234` to `2134` to `1234`): the flag
  fires. A shortest path does not fire it.
- Extract path on a hand-built selection produces a new strand whose members
  and order equal the selection.
- Presets still pass the Phase 3 gate after this work (regression run).

---

## Phase 5: The 1D transformation laws, generated for CM, VM, TM

Form: `D_I Phi_i = (L_I)_i^j Psi_j` and
`D_I Psi_j = i (R_I)_j^i d_tau Phi_i`. Generated from
`src/four_color/`, never transcribed.

**Precondition:** Phase 1 gate passed (this phase is independent of Phases
2-4 and may proceed in parallel with them; it must not merge into the
visualizer before Phase 7).

**What already exists, do not rebuild:** `src/four_color/` reproduces Tables
7-13 of arXiv:2408.09342 at the matrix level with tests. `cm.rs`, `vm.rs`,
`tm.rs` hold the L/R sets and the X/Y/Z/W hopper recursion; the convention is
locked by the Garden-algebra test (`src/four_color/mod.rs:159-165`). Two
typos in the published paper are already proven in code and must surface as
cited errata, never silently corrected:

- Table 7 Chiral L1 prints `<1 4 -2 -3>`; that value fails the Garden algebra
  and the internally consistent value is `<1 -4 2 -3>`
  (`src/four_color/cm.rs:21-32`, test at `src/four_color/mod.rs:181-193`).
- Table 12 Tensor R4 prints `<3 -2 4 3>`, which is not a permutation; the
  corrected value is `<3 -2 4 1>` (`src/four_color/mod.rs:196-204`).

**Work:** write the generator layer that turns the existing L/R matrices into
rendered equations (HTML or LaTeX), one block per multiplet per color. No
coefficient may be typed by hand; every sign and index comes from
`cm::l_matrices()`, `cm::r_matrices()` and the vm/tm counterparts.

**Exit gate:** the reproduction is the proof of correctness.
- Round trip: parse the generated equations back into signed permutation
  matrices; the parsed set must equal the source matrices exactly, for all
  three multiplets, all four colors, L and R.
- Table comparison: the generated coefficients match Tables 7-12 of
  arXiv:2408.09342 entry for entry, with exactly the two documented typo
  deviations and no others. Any third deviation fails the gate.
- `cargo test four_color` passes.
- Falsifiability proof: flip one sign in a copied L-matrix, round trip and
  table comparison both fail; restore, both pass.

---

## Phase 6: The same generator for VM1, VM2, VM3

**Precondition:** Phase 5 gate passed. The convention pinned by reproducing
CM/VM/TM is what makes the VM1-3 output defensible.

**Finding that reshapes this phase (resolved during planning):** the claim
that no VM1/VM2/VM3 matrices are published is false as an absolute. Signed
L-matrix representatives for all three are published in arXiv:1210.0478
(Chappell, Gates, Hubsch), Appendix B:

- VM1, Eq. (B.1): `(6)b<1432>, (3)b<2341>, (10)b<3214>, (0)b<4123>`
- VM2, Eq. (B.2): `(12)b<1243>, (9)b<2134>, (0)b<3421>, (10)b<4312>`
- VM3, Eq. (B.3): `(12)b<1234>, (5)b<2143>, (0)b<3412>, (6)b<4321>`

What remains unpublished, in every locally held Gates paper, is the full
Table 7-12 style treatment for VM1-3 (signed L AND R sets with the X/Y/Z/W
tower). So this phase produces something new, but it also has a published
gold standard to check one signing of each set against. The delivered
visualizer data already references these Appendix B signings
(`template:237`, "Appendix B #"), so the ingestion path exists.

**Work:**
1. Run the Phase 5 generator on VM1, VM2, VM3, using the quartet supports of
   arXiv:1210.0478 Eq. (5.2d-f) and the sign convention pinned in Phase 5.
2. Compare the generated signing against the Appendix B representative for
   each multiplet. Match or no match, record the outcome; both are valid
   Garden signings and a mismatch is a convention difference, not an error.
3. State the convention caveat visibly in the output: every quartet admits
   256 Garden signings, so 256 valid choices exist, not one canonical
   answer. Matching the published convention on CM/VM/TM pins the
   convention; applying that rule to VM1-3 is defensible, not unique.

**Exit gate:**
- Garden algebra check (`four_color::garden_ok`) passes for all three
  generated sets.
- Permutation supports equal Eq. (5.2d-f) exactly, verified on VM1 (quartet
  6) and VM2 (quartet 2) member by member. VM3 is quartet 1, the case that
  cannot fail; it does not count as verification.
- The Appendix B comparison is executed and its outcome (match or documented
  convention difference, per multiplet) appears in the output.
- The 256-signings caveat text is present and visible in the rendered
  output, not in a comment.

---

## Phase 7: Equations in the visualizer, as a drill-down

**Precondition:** Phases 4 and 6 gates passed.

Gates called an earlier artifact a cluttered "blurry ball", so clutter is a
known sensitivity. The equations must be reachable, never ambient.

**Work:** per-strand drill-down (click a strand's multiplet identity to open
its transformation laws, generated in Phases 5-6). Nothing renders until
asked. CM/VM/TM blocks cite arXiv:2408.09342 Tables 7-12 with the two typo
errata; VM1-3 blocks cite arXiv:1210.0478 Eq. (5.2), Appendix B, and carry
the 256-signings caveat.

**Exit gate:**
- With every drill-down closed, the rendered canvas is pixel-identical to
  the Phase 4 build (image hash comparison of the default view).
- Equations appear only after an explicit click, and close fully.
- The typo errata and convention caveat are visible inside the drill-down
  for the multiplets they apply to, checked on VM1 and VM2 specifically.
- Regression: Phase 3 and Phase 4 gates re-run and pass.

---

## Erratum disclosures (preserved; do not drop)

**VM2 quartet.** In the demoed image `3412` sits directly above `4312` in the
VM2 neighborhood, and arXiv:2408.09342 Tables 3 and 5 print `3412` for VM2.
He could confirm his own erratum off the wrong row. The tool's data is
already correct (`3421`); only the disclosure is missing. The visible note
(Phase 2, item 7): VM2 is shown as `3421`; the printed `3412` belongs to VM3
and admits no Garden signing (0 of 65,536), while `3421` admits 256, matching
all five other quartets. The correct VM2 support is confirmed independently
by arXiv:1210.0478 Eq. (5.2e).

**Two further typos in arXiv:2408.09342,** proven by tests in
`src/four_color/mod.rs:181-204` and surfaced in Phase 5: Table 7 Chiral L1
and Table 12 Tensor R4 (details in Phase 5). Whether and how to flag these to
Gates is an open question below.

---

## Explicitly excluded

The 4D N=1 transformation laws transcribed from a screenshot of his screen
share are excluded from every phase. A prior transcription of them contained
three errors. If they ever appear in any artifact, they appear as a cited
quotation with the source visible (arXiv:2010.14659 Appendix A, conventions
from arXiv:0902.3830 Appendix A), never as original work. Nothing in Phases
5-7 depends on them; the 1D laws are generated from matrices in this repo.

---

## Open questions

For Gates, batched so each call spends his time well:

1. **Entry node.** The preset animation enters each quartet at position 1 and
   retraces cumulatively, as he demoed (L11-15, L43-55). Confirm entry at
   position 1 versus entering at the quartet's base-hexagon node.
2. **The six steps.** He called the permutahedron "step one" and said
   "there's six steps in general" (SL65-67) but never enumerated the rest.
   What are steps two through six?
3. **The three paper typos.** VM2 `3412`/`3421` in Tables 3 and 5, Chiral L1
   in Table 7, Tensor R4 in Table 12 of arXiv:2408.09342. Does he want these
   passed to his coauthor, or kept as footnotes in the tool?
4. **Howard page 61 garble check.** His spoken base-face list (L179) includes
   "2, 3, 4, 1", almost certainly a garble of `2314` (all base-face members
   end in 4). Confirm against page 61 directly.

For Brandon, not for Gates:

5. **PrM1-PrM4 link identities.** The atlas passage ("In the first two, we
   give a more or less complete 'atlas' of how 4-color SUSY multiplets are
   mapped into the 4-color permutahedron") is from Gates's email of
   2026-07-20, not from a paper. The four PrM references are hyperlinks whose
   URLs did not survive the plaintext paste preserved locally. Recover the
   URLs from the original email. Best local candidates: arXiv:1210.0478 and
   arXiv:2012.13308 for the 4-color pair, arXiv:2304.09830 among the
   40,320-node pair. Nothing in Phases 5-6 depends on the answer anymore:
   the load-bearing sub-question (are VM1-3 signed matrices published?) is
   resolved above from arXiv:1210.0478 Appendix B.
6. **Howard PDF pages 61 and 64.** Permission-blocked from the planning
   environment; both must be opened manually for the Phase 3 thickness
   comparison and the question 4 check.

---

## Appendix: why the ordering error survived four review passes

The quartet `1234, 2143, 3412, 4321` is identical under ascending weight,
under final-digit ordering, and under hopper ordering. It is also the quartet
anyone naturally spot-checks, being the Vierergruppe itself. Three separate
errors (the ordering, the height key, the false captions) all passed review
because the one case that was checked is the one case that cannot fail. The
structural fix is ground rule 3 (verify quartets 4 and 6, never cite quartet
1 as evidence) and ground rule 2 (assertions recomputed from rendered
geometry, each proven falsifiable once). The same failure had a second root:
`selfCheck()` read conclusions the build script had already baked in, so the
page could not disagree with its own inputs. Checks that share provenance
with the thing they check are decoration, not verification.
