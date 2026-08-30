# Lever A, shard level: slot-0 orbit structure of the CLS L m=3 census (measured)

Date: 2026-08-18. Engine: `gmatrix_csp.v2-ac3` (unmodified; zero engine changes were
needed for this analysis). All claims below are computed, not inferred; every script
is reproducible from the facts recorded here.

## Summary

The 825 shard items of the L-side m=3 census collapse to **116 orbits** under a
measured symmetry action. Running one canonical item per orbit and replicating
counts to orbit-mates by a proven bijection reduces the L-side census from
~3.34B nodes (~1,435 core-hours) to ~374M nodes (~161 core-hours, ~8h wall on
stonkbot at T32). A block-diagonal signed permutation P with
P A_L P^-1 = A_R was verified entrywise, so the R-side census is the image of
the L-side census under a computable bijection: the R side costs nothing beyond
artifact generation. Combined: roughly an 18x reduction against the two-sided
fleet plan.

## The symmetry group (measured)

- A_L is block diagonal with three 4x4 blocks B0, B1, B2. The blocks are
  pairwise DISTINCT as matrices but pairwise signed-perm-conjugate (24
  conjugators between any ordered pair, including L-side to R-side pairs).
- Each block's signed-permutation normalizer N(B_i) has order 24: permutation
  parts are exactly the 12 even permutations (A4), each realized with 2 sign
  patterns that are negatives of each other.
- Full signed-perm normalizer of A_L (m=3): |N| = 24^3 x 6 = 82,944 (wreath
  product over S3 block permutations with sign-carrying conjugators). The
  global -I conjugates trivially on every G, so the effective order is 41,472.
- COMPLETENESS PROVEN: an independent exhaustive backtracking search over
  signed permutations with no wreath assumption (scripts/lever_a/group_n.py,
  `all_signed_normalizers`) returns set-equal groups at m=2 (1,152), m=3
  (82,944), and the block-0 stabilizer (27,648). So these are the complete
  signed-permutation normalizers, not just constructed subsets.
- m=2 (top-left 8x8): 1,152 elements, every one verified elementwise tonight
  to satisfy P A P^-1 = A.

## Slot-0 semantics (from the engine source)

The engine searches g = top-left 6x6 of P^-1 G P (P = block-diagonal eigenbasis
change, `build_coords`, gmatrix_full.rs:408), g^2 = lambda I_6 over
K = Q(sqrt(-3)), slots = 3x3 grid of 2x2 K-blocks (the 9-bit support masks).
Slot 0 fixes exactly the leading 4x4 integer block G[0..4][0..4], injectively:
825 surviving slot-0 values (`canonical_run`, gmatrix_csp.rs:2053, BTreeSet
order) = 825 distinct leading blocks; item index = ascending list position.

## The induced action on shard items

Only group elements fixing block position 0 map the slot-0 fiber of one shard
onto the slot-0 fiber of another. That stabilizer (order 27,648) acts on the
slot-0 value V (a 4x4 integer block) as V -> S V S^-1 with S in N(B0); the two
sign patterns per permutation cancel under conjugation, so the effective action
is the 12 A4 maps.

Measured orbit partition of the 825 items:

| orbit size | count |
|---|---|
| 1 | 9 |
| 4 | 48 |
| 6 | 14 |
| 12 | 45 |

Closure: 0 violations. Every image of every one of the 825 values under all 24
maps is again one of the 825 values.

## Validation against the stride-100 anchors

The 9 measured shards (results/cls_g_csp_shards_L_3blocks_s100) are the ground
truth. Orbit-mate shards must have identical solution counts AND identical node
counts (the action is a bijection on solution sets and a search-tree
isomorphism).

- Items 200 and 600 land in one orbit (size 12). They are the ONLY pair among
  the 9 with identical count (135,936) and identical nodes (5,828,587).
- The four empty anchors (100, 300, 700, 800) sit in distinct orbits, matching
  their four distinct node counts (783,979 / 755,179 / 719,467 / 811,627).
- Item 0 (the heavy outlier, 14,010,043 nodes) is in a size-6 orbit.

VALIDATION: PASS.

Work-weighted reduction from the anchors: 8.94x on nodes (unweighted item
factor 7.11x). Eight of the 116 orbits are already fully measured.

## Consequences for the census

- L side: run 116 canonical items (~108 not yet measured), ~374M nodes,
  ~161 core-hours, ~8h wall on stonkbot alone (13,203 nodes/s at T32), faster
  with the M4 enlisted.
- R side: P A_L P^-1 = A_R verified entrywise (block-diagonal signed perm
  assembled from per-block conjugators). The R solution set is the image of L
  under G -> P G P^-1. The R census needs no search.
- Total count: exact, as sum over orbits of orbit_size x count(rep). Emptiness
  propagates: the 4 empty anchors imply 40 of 825 items are empty (their orbit
  sizes are 12, 12, 12, 4).

## Open deliverable question

Skipped shards get exact replicated counts (bijection-proven) but no per-shard
checksums, because shard files do not store solutions and the splitmix64-sum
checksum is not equivariant under the group action. The merged artifact's
semantics become "116 canonical shards plus orbit map" unless a full
enumeration with per-shard checksums is required. Optional belt-and-braces:
spot-run a few skipped shards and compare replicated vs measured counts.

Per-node in-tree symmetry breaking is NOT needed for the count. It would only
matter if full per-shard checksums over all 825 items are required, in which
case there is no search saving anyway.

## Artifacts

- results/lever_a_slot0_orbits_L_3blocks.json: orbit map, sizes, anchor
  validation, group facts, L->R conjugator.
- scripts/lever_a/group_n.py: self-verifying group module (pure Python,
  stdlib only; full check suite runs in ~2s, group caches rebuild in <1s so
  no large cache files are kept).
- scripts/lever_a/slot0_items.json: the 825 slot-0 values (index, 4x4
  entries, K-slot encoding), dumped from the engine via a temporary in-crate
  test (added, run, reverted; git diff clean).
- scripts/lever_a/slot0_semantics.md: engine-semantics writeup with source
  line references.
- Anchor data: results/cls_g_csp_shards_L_3blocks_s100/shard_*.json.

## Fleet launch (2026-08-19)

Engine extension: `cls-g-csp-shard-items [side] [blocks] [items] [threads]
[dir]` on a clean worktree (`/Users/brandon/code/adinkra-codespace-itemspec`,
branch shard-items-spec, commit 7d9d9d1, base 61228b4 pre-Lever-B since Lever
B was measured speed-neutral and the main tree is dirty with 11D work).
`run_shards` was split into `run_shards_core` with an `ItemSelection` enum
(Window = original CLI semantics preserved byte-for-byte; Explicit = item
list). Validation: m=2 census via both forms merged to count 15000 /
checksum 94e85bc1c8e786fd / nodes 15211 (exact recorded ground truth; the
only inter-run differences are `seconds` timing fields), csp_m2 regression
test green, m=1 = 12 / 4b3aa9ef562965aa on both M4 and stonkbot (x86
neutrality), out-of-range item exits 2, non-increasing spec / empty entry /
threads=0 exit 1, shared-dir manifest accepted with resume-skip.

Assignment (scripts/lever_a/fleet_assignment_canonical116.json): 116 reps
sorted, positions mod 4; stonkbot 87 items at T32 (13.3k n/s measured after
launch), macm4 29 items at T12 (~4.4k n/s). Both run dirs seeded with the 9
stride-100 anchors; reps 0 and 100 were adopted by resume-skip (14.8M nodes
free), and the anchors double as join-time cross-checks.

Observability: heartbeat 15s, per-pod status_*.jsonl, and
scripts/lever_a/fleet_watch_canonical116.sh (polls both pods, emits on item
count change, 20-minute ETA summaries against the ~374M-node projection,
final banner when both report SHARDS DONE).

Join: scripts/lever_a/join_canonical_census.py collects rep shards from one
or more run dirs (rsync stonkbot's dir back first), structurally validates
each, refuses on count conflicts between duplicate measurements, replicates
counts/nodes across orbit-mates, cross-checks every measured anchor, and
writes results/four_color_cls_gmatrix_csp_L_3blocks_orbit_census.json with
exact totals, replicated class histogram, and canonical-only checksums.

Third machine: scripts/lever_a/fleet_assign_next.py <pod> <threads> prints
the paste-ready command over stonkbot's undone items taken back-to-front
(stonkbot grinds front-to-back; overlap is wasted compute, never wrong
results, because items are claimed only by a validating shard file).

## Census completed (2026-08-21)

The canonical-116 run finished with item 52 (the effort record: 78,070,315
nodes, 3,437,976 solutions, 81,267 s on one effective M4 core; the seed has 8
nonzeros, exactly 2 per row and column, the sparsest non-degenerate seed in the
census). The join over the macm4 canonical dir plus the stonkbot (93 shards)
and b300 (6 shards) mirrors reports:

- canonical shards: 116/116 present
- **L m=3 total count: 123,984,864**
- classes: 1,076
- anchor cross-checks: 7, verdict PASS
- canonical node total 421,625,916; replicated equivalent 3,436,350,339
  (8.15x work-weighted saving, vs ~8.1x projected above from the orbit map)

Final orbit-mass shares (total 123,984,864): size 1: 7,970,112 (6.4%),
size 4: 13,252,608 (10.7%), size 6: 17,891,712 (14.4%), size 12: 84,870,432
(68.5%). Item 52's orbit alone is 41,255,712, 33.3% of the census.

Class taxonomy vs orbit coupling tightened: 854 of 1,076 classes (79.4%)
appear in exactly one orbit size (was 73% at 115 orbits). Item 52 contributes
298 classes, 268 of them unseen in any other orbit; only 30 shared.

Artifacts: results/four_color_cls_gmatrix_csp_L_3blocks_orbit_census.json
(final census), results/cls-12x12-orbit-table-20260820.md (116 rows),
results/cls-12x12-census-onepager-20260820.md (headline summary). R side
identical by the verified conjugator; artifact generation only.
