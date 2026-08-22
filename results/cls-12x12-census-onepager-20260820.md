# CLS 12x12 G-matrix census: exact final results (L side), 2026-08-20

Engine: `gmatrix_csp.v2-ac3`. Method: 116 canonical orbit reps measured exactly
across a 3-machine fleet; counts replicated to orbit-mates by the proven
S V S^-1 bijection (Lever A, closure zero violations). Validation: stride-100
anchors cross-checked (join verdict PASS, 7 checks), duplicate machine
measurements agree across macm4/stonkbot/b300, join refuses on any conflict,
and the independent G -> -G solution bijection pairs 106 orbits into 53
count-equal pairs (10 self-negative orbits), all agreeing exactly.

## Headline numbers

- **Exact L-side total: 123,984,864** (116/116 orbits measured and joined)
- R side: identical totals, no second search (P A_L P^-1 = A_R verified entrywise)
- Items covered: 825 of 825 (709 exact by replication, 116 direct)
- Search work measured: 421,625,916 nodes over the 116 reps; full-825 equivalent
  3,436,350,339 nodes, so orbit reduction saved 8.15x work-weighted
- Distinct solution classes: 1,076

## Orbit structure

| orbit size | orbits | replicated mass | share |
|---|---|---|---|
| 1 | 9 | 7,970,112 | 6.4% |
| 4 | 48 | 13,252,608 | 10.7% |
| 6 | 14 | 17,891,712 | 14.4% |
| 12 | 45 | 84,870,432 | 68.5% |

Orbit sizes are exactly {1,4,6,12} (all divisors of 12, A4 action). The
count-crown item 204 (all-zero seed, 7,430,976) is one of the 9 singleton
orbits. Item 52's size-12 orbit alone carries 41,255,712 matrices, 33.3% of
the census.

## The record search: item 52

Item 52 (slot-0 seed: 8 nonzeros, exactly 2 per row and column) was the
heaviest search in the census and the last to land:

- count: 3,437,976 (2nd largest; item 204 keeps the count crown at 7,430,976)
- nodes: 78,070,315 (census effort record, 3.6x item 204's 21.6M)
- time: 81,267 s (~22.6 h) on one effective core of the M4
- density: 22.7 nodes per solution (vs 2.9 for item 204): a genuinely hard
  tree, not a wide one
- classes: 298, of which 268 appear in no other orbit

The seed-sparsity law held: all-zero seed (204) 21.6M nodes, 2-regular sparse
seed (52) 78.1M nodes, dense seeds finish under 3M. Sparse leading blocks give
the propagator nothing to bite on, so the tree stays narrow and deep.

## Class structure (1,076 distinct classes, 116 orbits)

- **79.4% of classes (854/1,076) appear in exactly one orbit size**: the A4
  orbit partition and the (nnz, support, ranks) class taxonomy are strongly
  coupled (73% at 115 orbits, rising as the sparse orbit landed).
- Item 52's orbit contributes 268 brand-new classes, 24.9% of census class
  diversity from a single orbit; only 30 of its classes are shared.
- The top classes by mass form one family: full support (111111111), all ranks
  4, nnz 64-96, spread across sizes 4/6/12.
- All-rank-4 classes with a single demoted rank are exclusive to singleton
  orbits.

## Cost accounting

- Marginal cash cost: $0 (stonkbot and macm4 are local machines; b300 cycles
  were borrowed from an existing pod paid by the 11D workstream; the pencilled
  $75-85 hybrid pod fleet was never launched, Lever A made it unnecessary)
- Nodes measured: 421,625,916 canonical-rep nodes fleet-wide
- Wall time: ~2 days across stonkbot (T32), M4 (T12), b300 (T8); the final
  item ran 22.6 h single-tree on the M4 after stonkbot was pulled from the race
- Measured item-weight distribution: median 2.17M, p90 6.52M, effort record
  78.1M (52), count record 7.43M (204): heavy-tailed, sparse-seed driven
- Orbit reduction: 825 items to 116 measured = 8.15x work-weighted saving

## Artifacts

- results/four_color_cls_gmatrix_csp_L_3blocks_orbit_census.json (final census)
- results/lever_a_slot0_orbits_L_3blocks.json (orbit map, group facts)
- results/cls-12x12-orbit-table-20260820.md (116-row table, complete)
- results/lever_a_v4_bridge_L_3blocks.json (V4/permutahedron bridge check,
  2026-08-21, script scripts/lever_a/v4_permutahedron_bridge_check.py)
- Run dirs: results/cls_g_csp_shards_L_3blocks_canonical (macm4 + anchors),
  stonkbot mirror (93 shards), b300 mirror (6 shards)
