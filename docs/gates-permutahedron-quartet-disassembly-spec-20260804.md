# Technical spec: S4 permutahedron six-quartet disassembly animation (Gates call 2026-08-04)

Source: `~/Downloads/Gates Permutahedron chat 20260804.txt` (auto-transcript of a
Zoom call, S. J. Gates requesting, collaborator implementing). Three-agent
line-by-line extraction, synthesized and mathematically cross-validated here.
Every requirement below carries the transcript line numbers it came from.
Artifact under revision: the six-quartet disassembly HTML
(`S4_Six_Quartet_Disassembly.html`, 2026-08-03) plus its successor animation.

Transcript garble dictionary (used throughout): "communichedron" =
permutahedron (line 9); "dinkers" = adinkras (line 259); "the Bermudahedron" =
Gates's name for the quartet-listing figure, Howard talk page 61 (line 199);
"G1" = a Gates paper file containing the CM/VM/TM equation figures (lines
227-233); "identity matrix" = the identity permutation 1,2,3,4 (line 63).

---

## 1. Mathematical substrate (decoded and cross-validated)

### 1.1 The object
- The S4 permutahedron: 24 vertices, one per permutation of (1,2,3,4);
  truncated octahedron with 6 square faces and 8 hexagonal faces.
- Generators are the three adjacent transpositions, rendered as colored edges:
  red, blue, green. Explicitly witnessed in the transcript:
  - blue swaps positions 1-2: 1,2,3,4 -> 2,1,3,4 (line 19),
  - red swaps positions 3-4: 2,1,3,4 -> 2,1,4,3 (line 21),
  - green is the remaining generator (implied, lines 29, 65).
- Square faces come from commuting generators: blue and red commute
  (s1 s3 = s3 s1), stated and verified on a face at lines 65-69: "for square
  faces, whether you go blue first or red... it doesn't matter... That's true
  on any square face."
- Hexagonal faces come from non-commuting adjacent generator pairs.
  The operative hexagonal face is the one containing the identity (lines 43,
  63, 175, 177).

### 1.2 The six quartets (structural decode, consistent with every tuple in the transcript)
- The identity's quartet is {1,2,3,4 ; 2,1,4,3 ; 3,4,1,2 ; 4,3,2,1}
  (lines 11-35, consolidated at line 185). These are exactly the Klein
  four-group V4 = {e, (12)(34), (13)(24), (14)(23)} acting on positions.
- The six quartets are the six right cosets of V4 in S4 (24 = 6 x 4).
- Each coset contains exactly one permutation with value 4 in position 4.
  Those six permutations are the S3 subgroup {1234, 2134, 1324, 2314, 3124,
  3214}, which is precisely the vertex set of the hexagonal face through the
  identity. This is the structural content of Gates's rule "all the starting
  nodes... lie on this hexagon face" (line 177).
- The six starting members ("bottoms" of the chains), as dictated at line 179:
  1,2,3,4 ; 1,3,2,4 ; 3,1,2,4 ; 3,2,1,4 ; 2,3,4,1 ; 2,1,3,4.
  NOTE: 2,3,4,1 (2341) does not fix position 4 and cannot be a coset bottom;
  it is almost certainly a transcript garble of 2,3,1,4 (2314). The other five
  match the S3 subgroup exactly. Verify against Howard talk p.61 before
  shipping; do not silently re-sort or substitute.
- Display order within each quartet chain: the bottom (position-4-fixing
  member) first, then the remaining three members in ascending numeric order
  (labels read as 4-digit integers, lines 183-187).
  - Worked example 1 (identity chain): 1234, then 2143 < 3412 < 4321
    (line 185).
  - Worked example 2 (the 3214-bottom chain, coset {1432, 2341, 3214, 4123}):
    bottom 3214, then 1432 < 2341 < 4123. Gates dictates "1,4,3,2... 2,3,4,1
    should be the next member" at line 189, matching exactly.
- "Bottom/lowest member" is a layout term (the chain's base on the page), not
  the numeric minimum: 3214's chain displays 3214 first even though 1432 is
  numerically smaller (lines 47, 87, 179 vs 189).

### 1.3 The journey/path system
- From each bottom, the other three quartet members are reached by "journeys"
  along permutahedron edges (lines 11-35).
- Nested prefix structure: journey n restarts at the bottom and retraces the
  full journey n-1 path before extending (lines 11, 15, 23, 31, 33).
- Governing rules (lines 147-161, 181): greedy minimum link count at each
  step, never backtrack; Gates names this the "self-avoiding random walk"
  (line 161).
- Canonical example (identity bottom, from lines 19-35):
  - Journey 1: 1234 -[blue]-> 2134 -[red]-> 2143 (member 2). Two edges.
  - Journey 2: retrace, continue 2143 -> 2413 -> 2431 -[?]-> ... reaching
    4321 (member 3) after 4 links beyond 2143 (line 155), total path
    1234..4321 of length 5-6.
  - Journey 3: full replay (five glowing edges, line 33) then one more edge
    4312 -> 3412 (member 4, line 35).
  - Full traversal node sequence: 1234, 2134, 2143, 2413, 2431, 4321, 4312,
    3412.
- Another witnessed path (3124 bottom, to its partner 4213): color sequence
  red, green, blue, red, green, red, six links (lines 87-95).
- The path system is the same from every bottom ("the paths I'm taking are
  essentially the same paths", line 57) and is given in the Howard talk as
  permutations in cycle notation (line 57). Howard talk p.61 is the ground
  truth path table (lines 199-203).
- KNOWN DEFECT in transcript colors: color labels for edges into 2431 and 4321
  (line 29) and the R,G,B,R,G,R sequence (line 93) are not consistent with the
  position-transposition convention of lines 19-21. Record them as reported,
  but the animation's edge-color data must be generated from the Howard talk
  p.61 table (or computed from the V4-coset structure and verified against
  p.61), not from the raw transcript.

---

## 2. Corrections ledger (defects in the 2026-08-03 artifact that must be fixed)

C1. Quartet member ordering is wrong (the headline issue).
    Members must be: bottom first, then ascending numeric (lines 143-157,
    185-191, 201, 211). Specific swaps dictated: 4,3,2,1 before 3,4,2,1
    (line 151); chain rebuild for bottom 2,1,3,4 (missing/reversed, lines
    171-173); "the order of these two nodes needs to be interchanged"
    (lines 143, 165). CONFLICT FLAG: line 143 as transcribed ("3,4,1,2 should
    be first, 2,1,4,3 second") contradicts the ascending rule; it refers to
    on-screen positions Gates was pointing at. Resolve all ordering against
    Howard p.61, not against this line.
C2. No permutation may appear twice across the six chains (line 105).
C3. Black/white styling: white regions must carry black letters/numbers
    (line 165: "White includes in black letters or black numbers" [garbled,
    intent: contrast fix]). Acknowledged as simple (line 223).
C4. Path representation: the previous version "had all the same color
    together"; the corrected version must encode the actual colored journey
    paths (lines 39-41).
C5. Disassembly must be strand by strand, not all strands at once (line 211).
    The rendered permutahedron image itself is fine and stays (line 211).

---

## 3. Animation and interaction requirements

R1. Journey rendering: each journey animates as sequential edge-glowing on
    the permutahedron; the traversed link glows as it is crossed, including
    every retraced prefix edge (lines 15, 17, 25, 29, 31, 33).
R2. Pacing: after each journey completes, the animation pauses ("stop and
    rest"), then resets to the bottom and begins the next journey (line 23).
R3. Orientation: render the permutahedron with the hexagonal face (the one
    containing the identity) at the floor (line 177).
R4. Interaction model: click-through advance; each click performs the next
    segment. No auto-play; Gates narrates live over it (lines 213, 215).
R5. Disassembly: six pop-outs, in strand order 1 through 6, one per click;
    each strand's four nodes pull out of the polyhedron, move aside, and
    coalesce into the straight-line chain diagram (lines 37-39, 211-215).
R6. Vacated nodes: after a strand pops out, its former vertices on the
    permutahedron go blank, rendered as plain white balls; blanking
    accumulates across the six steps (line 219).
R7. Reassembly must follow the same ordering as disassembly (line 203:
    "disassembling and assembling, we want to follow this").
R8. Both the pop-out chains and the journey animation must use the Howard
    p.61 ordering (lines 201, 203).
R9. Lecture context: the artifact is lecture material; all pacing must be
    user-driven and robust to arbitrary dwell time between clicks
    (lines 203, 215).

---

## 4. Acceptance criteria / QA checklist

A1. The six chains partition all 24 permutations; each tuple appears exactly
    once (C2).
A2. Each chain starts at its position-4-fixing member; the six starts are
    exactly the vertices of the identity's hexagonal face (1.2).
A3. Within each chain, members after the first are in strictly ascending
    4-digit numeric order (A = "never descending", line 187).
A4. The chain listing is identical to Howard talk p.61 ("The Bermudahedron"),
    which is the declared ground truth (lines 199-201).
A5. Every displayed journey path is shortest-path, self-avoiding, and uses
    the nested-prefix structure (1.3).
A6. Six click-through pop-outs in order, white-ball vacating, straight-line
    coalescence, strand-at-a-time (R4-R6).
A7. Black numbers on white regions (C3).
A8. QA admission on record: the prior version shipped without this checking;
    the ordering reference "was right here the whole time" (lines 207-209).
    Verify A1-A4 programmatically against a generated V4-coset table before
    delivery, then eyeball against p.61.

---

## 5. Research-program context (where this artifact sits)

- CM, VM, TM (chiral, vector, tensor multiplets) each correspond to a distinct
  set of differential equations, shown from paper G1 (lines 253-257).
- Adinkras are "graphical fingerprints" of those equations; equation
  properties are to be tracked by adinkra shape, "the whole point of the
  project" (lines 257-259). Some target equations have never been found
  (line 261); the goal is to map them all (line 263).
- The four-color S4 permutahedron (this artifact) is the base case tied to
  the CM/VM/TM equation sets (line 265).
- Next big target: the omni-truncated 7-simplex, "the hex," with "40,000-
  something" nodes per Gates (line 263; structurally 8! = 40,320 vertices,
  the S8 permutahedron). The lecture materials being built now are the S4
  template for that campaign.
- Delivery/admin: the collaborator committed to delivery tomorrow ("probably
  be looking for it tomorrow", line 273, spoken 2026-08-04, so due
  ~2026-08-05), with a fallback delivery channel if the file does not come
  through (line 275).

---

## 6. Open items and ambiguities (resolve before shipping)

1. Line 179 start list: "2,3,4,1" vs structurally required "2,3,1,4".
   Verify against p.61.
2. Line 143 swap direction conflicts with the ascending rule; treat p.61 as
   ground truth (see C1).
3. Edge-color naming in journey segments (lines 29, 89, 93) is inconsistent
   with the lines 19-21 convention; generate color data from structure and
   verify against p.61 (see 1.3).
4. Line 191: third and fourth members of one chain were pointed at, not
   spoken; unrecoverable from the transcript, get from p.61.
5. Line 105's trailing "and 4" is garbled; that quartet's membership was cut
   off mid-sentence.
6. Line 89's "green, red" second-member path vs line 65's "blue, red" claim;
   reconcile via p.61.
7. "Bermudahedron" and "G1" need their actual files identified in Gates's
   materials for citation in the artifact's provenance notes.
