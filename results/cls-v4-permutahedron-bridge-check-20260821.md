# Census A4 orbits vs permutahedron V4 quartets: bridge check, 2026-08-21

Question: does the completed CLS 12x12 census orbit structure meet the S4
permutahedron quartet structure from the Gates 2026-08-04 spec
(docs/gates-permutahedron-quartet-disassembly-spec-20260804.md), and does the
permutahedron geometry (identity hexagon face, quartet bottoms) pick out any
special stratum of the census?

Method: scripts/lever_a/v4_permutahedron_bridge_check.py (stdlib only, reuses
the self-checked group_n.py normalizer machinery, rebuilds the orbit partition
from raw block data before comparing to the orbit map). Machine-readable
output: results/lever_a_v4_bridge_L_3blocks.json. All checks PASS.

## The shared substrate

Both sides are governed by the chain S4 > A4 > V4 on 4 points:

- Permutahedron: 24 vertices, six quartets = right cosets of
  V4 = {e, (12)(34), (13)(24), (14)(23)}; quartet bottoms = position-4-fixing
  members = the identity hexagon face (an S3, whose even part is the C3 fixing
  point 4).
- Census: 825 slot-0 blocks, 116 orbits under the 12 A4 conjugators of N(B0)
  (sizes 9x1, 48x4, 14x6, 45x12).

## Finding 1: the V4 refinement is exact

Restricting the census action to V4 gives 234 V4-orbits with sizes
{9 of size 1, 42 of size 2, 183 of size 4}: exactly what the subgroup lattice
predicts from the A4 orbit sizes (V4 meets every C3 trivially; every order-2
subgroup of A4 lies in V4), with zero deviations. The census partition is
consistently V4-equivariant all the way down.

## Finding 2: the census is face-blind (and provably so)

Labeling stabilizers by permutahedron geometry: C3 stabilizers by fixed point
(the 4 hexagon-face directions; fix4 = the identity face), C2 stabilizers by
double transposition (the 3 nontrivial identity-quartet elements):

| stratum | blocks | exact mass | share |
|---|---|---|---|
| C3 fix1 / fix2 / fix3 / fix4 | 48 each | 3,313,152 each | 2.7% each |
| C2 2143 / 3412 / 4321 | 28 each | 5,963,904 each | 4.8% each |

Perfectly uniform in blocks AND in exact census mass. This is forced, not
accidental: A4 is transitive on the 4 points, orbit-mates share counts by the
proven bijection, and the data confirms the two consequences exactly
(48/48 size-4 orbits meet each C3 stratum once; 14/14 size-6 orbits meet each
double transposition twice). So the identity hexagon face and the quartet
bottoms have NO distinguished shadow in the census: anything that distinguishes
one face or one quartet direction must live in the odd (S4-only) half of the
permutahedron story, outside the census's even symmetry.

## Finding 3: the structure beyond A4 is the G -> -G negation duality

The only non-uniform census structure is a duality the A4 action cannot see:
(-G)^2 = G^2, so G -> -G is an exact solution bijection sending slot-0 block
V to -V. Verified from the data:

- -V is a slot-0 value for all 825 blocks; negation is an involution and
  commutes with the whole A4 action.
- On the 116 orbits it induces 53 count-equal pairs + 10 self-negative orbits;
  all 53 measured pairs agree exactly. This is an independent census
  cross-check (different mechanism from the A4 anchors and the duplicate
  machine runs) and it passes.
- The 9 A4-singleton blocks organize as 4 negation pairs + 1 self-negative:
  (21, 401) at 1,152; (80, 336) at 268,416; (132, 290) and (135, 285) at 0;
  and item 204, the all-zero seed, self-negative, carrying the count crown
  7,430,976.

Bonus consequence for future censuses: negation halves the measurement
requirement (63 of 116 orbits, a further ~1.8x on top of Lever A's 8.15x).

## Verdict for the quartet deliverable

The bridge is real but uniform: the census is exactly V4-equivariant, the
quartet core acts freely and evenly, and no permutahedron direction is
distinguished from the G-matrix side. For the Gates animation this is clean
supporting material rather than new geometry: the six-quartet structure is an
S4 statement, its even A4 half descends to the census as pure uniformity, and
the census's own extra structure (negation pairing) has no permutahedron
counterpart. State it as: "the exact G-matrix census is V4-equivariant with a
perfectly equitable stratification; the identity face is invisible to it."

Reproduce: `python3 scripts/lever_a/v4_permutahedron_bridge_check.py`
(runs in seconds, all checks self-contained).
