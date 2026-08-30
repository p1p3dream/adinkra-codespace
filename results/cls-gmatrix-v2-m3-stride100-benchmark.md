# CLS G-matrix v2 engine: m=3 stride-100 stratified prefix benchmark

Date: 2026-07-30. Engine: gmatrix_csp.v2-ac3 (src/four_color/gmatrix_csp.rs),
product-to-support map residual filter (ADINKRA_CSP_REFINE default 65536).
Run: `cls-g-csp-shard L 3 0 825 4 results/cls_g_csp_shards_L_3blocks_s100 100`
on the local 4-thread machine, 18703s wall, 9 of 825 slot-0 prefixes.

## Completed-prefix timing distribution

| shard | count | nodes | seconds |
|-------|-------|-------|---------|
| 0000 | 514,368 | 14,010,003 | 18,672.8 |
| 0100 | 0 | 755,179 | 2,023.3 |
| 0200 | 135,936 | 5,828,587 | 8,386.9 |
| 0300 | 0 | 783,979 | 2,193.1 |
| 0400 | 198,144 | 2,629,003 | 4,240.2 |
| 0500 | 89,856 | 5,067,179 | 8,132.9 |
| 0600 | 135,936 | 5,828,587 | 8,658.7 |
| 0700 | 0 | 719,467 | 2,043.3 |
| 0800 | 0 | 811,627 | 1,988.4 |

Sample totals: 1,074,240 solutions, 36,433,651 nodes, 56,340 core-seconds
(15.65 core-hours), aggregate ~647 nodes/s/core.

Structure of the distribution:
- Empty prefixes cost a nearly constant ~2,050s (4 samples, 1988-2193s, +-5%).
- Solution-bearing prefixes scale with yield: 4,240s (198k), 8,133s (90k),
  8,387s (136k), 8,659s (136k), 18,673s (514k).
- Prefixes 0200 and 0600 are an exact symmetry pair: identical count, nodes,
  and class histograms, distinct checksums (85cf08ffa2468db1 vs
  66a56b8ee78b1e2c). Distinct solution sets of identical structure, same
  phenomenon as the L/R class-distribution match at m=2.
- Prefix 0000 is the outlier (5.2h, 514k solutions). If it is structurally
  special rather than typical, the stratified projection below is
  conservative-biased in count and time.

## Projected total work (uniform stratified estimate, x825/9)

- Count: ~98.5M solutions (very wide error bars; 48% of the sample's
  solutions sit in one prefix).
- Nodes: ~3.34B.
- Compute: ~1,435 core-hours. At 96 cores ~15 wall hours; at 64 cores ~22h.
- Memory: 1.6GB RSS per process (shared product tables 1.18GB + support maps
  ~140MB), independent of thread count.

## Speedup evidence vs v1 baseline

- m=2: v1 searched 119,849,219 nodes; v2 searched 15,211 (identical census,
  count 15000, checksums 94e85bc1c8e786fd L / cb9c5392d14b813b R). 7,878x
  node reduction. v2 wall time 0.16s vs v1 212.7s (L) / 1419.2s (R).
- m=3: v1 baseline has searched 81.8B nodes in 55.7h (0.3-0.7M nodes/s/core)
  without completing. If the m=2 node-reduction factor transfers, v1's m=3
  total is ~26 trillion nodes (~2 years at current rate); v2 projects ~3.34B
  nodes (~1,435 core-hours). v2 per-node cost is ~600x higher (exact
  K-arithmetic in deep propagation), but total work is ~20x lower with a
  finishing horizon, and v2 is sharded, resumable, and classifies its output.

## Validation evidence

- m=1: count 12, checksum 4b3aa9ef562965aa, matches v1, threads 1 and 3.
- m=2: count 15000, checksums 94e85bc1c8e786fd (L) and cb9c5392d14b813b (R),
  node-identical (15,211) across threads 1/4/7 and across engines.
- Checksum construction identical to v1 (splitmix64-sum of hash_intmat over
  verified integer G); every leaf re-verified G^2 = A exactly before counting.
- Class histogram sums equal shard counts exactly (checked per shard).
- Sharding: immutable per-prefix JSON (tmp+rename), items.json manifest,
  resume-skip verified (825/825 skipped on rerun), merge reproduces the m=2
  census exactly from 825 shards, missing shard -> merge refuses, exit 2,
  no artifact written.

## RunPod recommendation inputs

Gate from the directive: durable shard recovery (demonstrated at m=2, shard
format unchanged for m=3) plus substantial measured speedup (above). Both
met. Recommended shape: single 64-96 core CPU pod, one process, all 825
prefixes as shards in one directory (prefixes 0..824, stride 1), resume on
restart for free. Expected wall time 15-22h. Long-tail risk is concentrated
in prefix 0000-class shards (up to ~5h each at 4-thread speed); with 64+
threads the tail hides inside the schedule, so dynamic subdivision is
optional, not required.
