#!/usr/bin/env python3
"""Bridge check: census A4 orbit structure vs permutahedron V4 quartet structure.

Two sides being bridged
-----------------------
Permutahedron side (docs/gates-permutahedron-quartet-disassembly-spec-20260804.md):
the 24 vertices of the S4 permutahedron split into SIX QUARTETS = right cosets
of V4 = {e, (12)(34), (13)(24), (14)(23)}; each coset's bottom is its
position-4-fixing member; the six bottoms form the identity hexagon face (the
S3 fixing position 4, whose even part is the C3 fixing point 4).

Census side: the 825 slot-0 blocks split into 116 orbits under the 12-map A4
action V -> S V S^-1, S in N(B0) (results/lever_a_slot0_orbits_L_3blocks.json).

Shared substrate: the chain S4 > A4 > V4 acting on 4 points.  This script
computes, from the raw block data and the verified normalizer machinery:

  1. the V4-refinement of the census partition (V4 orbits of the 825 blocks),
  2. the stabilizer subgroup of every block in A4, labeled by permutahedron
     geometry: C3s are labeled by their fixed point (the even part of the
     corresponding hexagon-face S3; fixed point 4 = the identity face),
     C2s are labeled by their double transposition (the three nontrivial
     elements of the identity quartet V4),
  3. mass per stratum (exact, since every item's count equals its orbit
     rep's count by the proven bijection),
and prints the bridge verdict: uniform strata mean the permutahedron geometry
does not pick out any special census direction; concentration means it does.

Run: python3 v4_permutahedron_bridge_check.py   (writes results JSON)
Stdlib only.  Group machinery imported from group_n.py (self-checked there).
"""
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import group_n as G  # noqa: E402

ORBIT_MAP = os.path.join(ROOT, "results", "lever_a_slot0_orbits_L_3blocks.json")
SLOT0 = os.path.join(HERE, "slot0_items.json")
SHARD_DIRS = [
    os.path.join(ROOT, "results", "cls_g_csp_shards_L_3blocks_canonical"),
    os.path.join(ROOT, "results", "cls_g_csp_shards_L_3blocks_canonical_stonkbot_mirror"),
    os.path.join(ROOT, "results", "cls_g_csp_shards_L_3blocks_canonical_b300_mirror"),
]
OUT = os.path.join(ROOT, "results", "lever_a_v4_bridge_L_3blocks.json")

checks = []


def check(label, ok, info=""):
    checks.append(ok)
    print("[%s] %s%s" % ("PASS" if ok else "FAIL", label,
                         ("  (%s)" % info) if info else ""))
    return ok


# ---------------------------------------------------------------- group side
A_L = G.load_A("L")
B0 = G.blocks(A_L)[0]
N0 = G.block_normalizers(B0)          # 24 signed perms of range(4)
check("|N(B0)| = 24", len(N0) == 24)

# Effective action: the two sign patterns per perm give identical conjugation.
rep_signs = {}
for p, s in N0:
    rep_signs.setdefault(p, s)
check("12 effective perms (A4)", len(rep_signs) == 12)

perms = sorted(rep_signs)             # the 12 even perms of range(4), 0-indexed
IDENT = tuple(range(4))


def perm_name(p):
    """One-line 1-indexed notation, e.g. '2143'."""
    return "".join(str(x + 1) for x in p)


def cycle_type(p):
    seen = [False] * 4
    cycles = []
    for i in range(4):
        if not seen[i]:
            j, ln = i, 0
            while not seen[j]:
                seen[j] = True
                j = p[j]
                ln += 1
            cycles.append(ln)
    return tuple(sorted(cycles))


dts = [p for p in perms if cycle_type(p) == (2, 2)]        # 3 double transpositions
tri = [p for p in perms if cycle_type(p) == (1, 3)]        # 8 three-cycles
check("A4 = 1 id + 3 double transpositions + 8 three-cycles",
      len(dts) == 3 and len(tri) == 8 and IDENT in perms)

V4 = [IDENT] + dts                                        # the identity quartet


def fixed_point(p):
    """The point fixed by a 3-cycle (its C3 subgroup's label)."""
    return [i for i in range(4) if p[i] == i][0]


# The 4 C3 subgroups, each labeled by its fixed point (0..3 => position 1..4).
c3_of_point = {}
for p in tri:
    c3_of_point.setdefault(fixed_point(p), []).append(p)
check("4 C3 subgroups, one per fixed point",
      sorted(c3_of_point) == [0, 1, 2, 3]
      and all(len(v) == 2 for v in c3_of_point.values()))

# ---------------------------------------------------------------- block side
items = json.load(open(SLOT0))
block_of = {e["index"]: tuple(tuple(int(x) for x in r) for r in e["entries"])
            for e in items}
check("825 slot-0 blocks, all distinct", len(block_of) == 825
      and len(set(block_of.values())) == 825)
item_of = {v: k for k, v in block_of.items()}


def act(idx, p):
    img = G.conj_block4(block_of[idx], p, rep_signs[p])
    return item_of.get(img)


# Closure and partition rebuild from raw data (independent of the orbit map).
bad = [(i, perm_name(p)) for i in block_of for p in perms if act(i, p) is None]
check("closure: all 825 x 12 images stay in the 825 set", not bad,
      "%d violations" % len(bad))

seen = set()
a4_orbits = []
for i in sorted(block_of):
    if i in seen:
        continue
    o = {act(i, p) for p in perms}
    seen |= o
    a4_orbits.append(sorted(o))
check("rebuilt A4 partition = 116 orbits", len(a4_orbits) == 116,
      "sizes %s" % dict(sorted(Counter(len(o) for o in a4_orbits).items())))

om = json.load(open(ORBIT_MAP))
map_orbits = {frozenset(o["members"]) for o in om["orbits"]}
check("rebuilt partition set-equal to the orbit map",
      {frozenset(o) for o in a4_orbits} == map_orbits)
orbit_of = {}
for o in a4_orbits:
    for i in o:
        orbit_of[i] = tuple(o)

# Rep counts (exact per item by the proven replication bijection).
rep_count = {}
for o in om["orbits"]:
    rep = o["rep_item"]
    for d in SHARD_DIRS:
        path = os.path.join(d, "shard_%04d.json" % rep)
        if os.path.exists(path):
            rep_count[tuple(sorted(o["members"]))] = json.load(open(path))["count"]
            break
check("counts resolved for all 116 orbits", len(rep_count) == 116)
mass_of = {i: rep_count[orbit_of[i]] for i in block_of}
TOTAL = sum(mass_of.values())
check("mass check: total = 123,984,864", TOTAL == 123984864, f"{TOTAL:,}")

# ------------------------------------------------------- stabilizer census
stab = {}
for i in block_of:
    stab[i] = frozenset(p for p in perms if act(i, p) == i)

label = {}
for i, st in stab.items():
    if len(st) == 12:
        label[i] = "A4"
    elif len(st) == 3:
        fp = fixed_point(next(iter(st - {IDENT})))
        label[i] = "C3_fix%d" % (fp + 1)
    elif len(st) == 2:
        tau = next(iter(st - {IDENT}))
        label[i] = "C2_%s" % perm_name(tau)
    else:
        label[i] = "e"

check("stabilizer order = 12 / |A4 orbit| for all 825 blocks",
      all(len(stab[i]) * len(orbit_of[i]) == 12 for i in block_of))

stab_counts = Counter(label.values())
print()
print("stabilizer strata over the 825 blocks:")
for k in sorted(stab_counts, key=lambda k: (-stab_counts[k], k)):
    print("  %-10s %4d blocks" % (k, stab_counts[k]))

# ------------------------------------------------------------- V4 refinement
v4_seen = set()
v4_orbits = []
for i in sorted(block_of):
    if i in v4_seen:
        continue
    o = {act(i, p) for p in V4}
    v4_seen |= o
    v4_orbits.append(sorted(o))
v4_sizes = Counter(len(o) for o in v4_orbits)
print()
print("V4 refinement: %d V4-orbits, sizes %s"
      % (len(v4_orbits), dict(sorted(v4_sizes.items()))))

# Predicted purely from A4 orbit sizes + subgroup lattice (V4 cap C3 = e, all
# order-2 subgroups of A4 lie in V4):
pred = Counter()
for o in a4_orbits:
    s = len(o)
    if s == 1:
        pred[1] += 1
    elif s == 4:
        pred[4] += 1          # V4 acts freely => transitively
    elif s == 6:
        pred[2] += 3          # stabilizer C2 <= V4 => V4-orbits of size 2
    else:
        pred[4] += 3          # free
check("V4 refinement matches the subgroup-lattice prediction",
      v4_sizes == pred, "%s" % dict(pred))

# Every size-2 V4 orbit's stabilizer is the tau labeling its C2 stratum.
ok = True
for o in v4_orbits:
    if len(o) == 2:
        i = o[0]
        tau = next(iter(stab[i] - {IDENT}))
        ok = ok and act(i, tau) == i and label[i] == "C2_%s" % perm_name(tau)
check("each size-2 V4 orbit is stabilized by its own tau", ok)

# ------------------------------------------------------------ cross-tabs
print()
print("C3 strata (by fixed point; fix4 = identity hexagon face direction):")
for fp in (1, 2, 3, 4):
    members = [i for i in block_of if label[i] == "C3_fix%d" % fp]
    m = sum(mass_of[i] for i in members)
    print("  fix%d: %3d blocks, mass %13s (%5.1f%%)" % (fp, len(members), f"{m:,}", 100 * m / TOTAL))

print()
print("C2 strata (by double transposition = identity-quartet element):")
for tau in dts:
    name = "C2_%s" % perm_name(tau)
    members = [i for i in block_of if label[i] == name]
    m = sum(mass_of[i] for i in members)
    print("  %s: %3d blocks, mass %13s (%5.1f%%)"
          % (name, len(members), f"{m:,}", 100 * m / TOTAL))

fixed_blocks = sorted(i for i in block_of if label[i] == "A4")
print()
print("the 9 A4-fixed blocks (singleton orbits): item, seed nnz, count")
for i in fixed_blocks:
    nnz = sum(1 for r in block_of[i] for x in r if x)
    print("  item %3d  nnz %2d  count %s" % (i, nnz, f"{mass_of[i]:,}"))

# Within each size-6 A4 orbit: do the three V4 pairs carry the three taus?
tri_orbits = [o for o in a4_orbits if len(o) == 6]
patterns = Counter()
for o in tri_orbits:
    taus = tuple(sorted({label[i][3:] for i in o if label[i].startswith("C2_")}))
    patterns[taus] += 1
print()
print("size-6 orbit tau patterns (three V4 pairs each): %s" % dict(patterns))

# C3 strata inside size-4 orbits: one C3 stabilizer type per orbit?
quad_orbits = [o for o in a4_orbits if len(o) == 4]
qpat = Counter()
for o in quad_orbits:
    fp = tuple(sorted({label[i][6:] for i in o}))
    qpat[fp] += 1
print("size-4 orbit fixed-point types: %d orbits, types %s"
      % (len(quad_orbits), dict(qpat)))

# ------------------------------------------------------------------ verdict
# ------------------------------------------------- the G -> -G negation duality
# (-G)^2 = G^2, so G -> -G is an exact solution bijection; on slot-0 blocks it
# sends V -> -V.  This lives OUTSIDE the A4 action, so it explains any pairing
# the A4 side cannot (starting with the singleton blocks above).
neg = {}
for i in block_of:
    nv = tuple(tuple(-x for x in r) for r in block_of[i])
    neg[i] = item_of.get(nv)
check("negation: -V is a slot-0 value for all 825 blocks",
      all(v is not None for v in neg.values()))
check("negation is an involution", all(neg[neg[i]] == i for i in block_of))
check("negation commutes with the A4 action",
      all(neg[act(i, p)] == act(neg[i], p) for i in block_of for p in perms))

# Induced involution on the 116 orbits; paired orbits must carry equal counts
# (an exact cross-check of the census independent of the A4 anchors).
neg_orbit = {o: tuple(sorted(neg[i] for i in o)) for o in map(tuple, a4_orbits)}
ok_counts, self_neg, pairs = True, 0, 0
for o in map(tuple, a4_orbits):
    no = neg_orbit[o]
    if no == o:
        self_neg += 1
    else:
        pairs += 1
    if rep_count[o] != rep_count[no]:
        ok_counts = False
        print("  COUNT MISMATCH: %s vs %s" % (o[0], no[0]))
check("negation-paired orbits carry equal counts", ok_counts,
      "%d self-negative orbits, %d paired orbits" % (self_neg, pairs))

fixed_names = {i: neg[i] for i in fixed_blocks}
print()
print("negation on the 9 singleton blocks (item -> negated item):")
for i in fixed_blocks:
    print("  %3d -> %3d %s" % (i, fixed_names[i],
                               "(self)" if fixed_names[i] == i else ""))

c3n = Counter(label[i] for i in block_of if label[i].startswith("C3_fix"))
c2n = Counter(label[i] for i in block_of if label[i].startswith("C2_"))
uniform_c3 = len(set(c3n.values())) == 1
uniform_c2 = len(set(c2n.values())) == 1
print()
print("VERDICT: C3 strata %s, C2 strata %s"
      % ("uniform (%d each)" % c3n["C3_fix1"] if uniform_c3 else
         "NON-UNIFORM %s" % dict(c3n),
         "uniform (%d each)" % c2n["C2_2143"] if uniform_c2 else
         "NON-UNIFORM %s" % dict(c2n)))

out = {
    "source": "scripts/lever_a/v4_permutahedron_bridge_check.py",
    "engine": "gmatrix_csp.v2-ac3",
    "blocks": 3,
    "n_items": 825,
    "a4_orbits": dict(Counter(len(o) for o in a4_orbits)),
    "v4_orbits": {"count": len(v4_orbits), "sizes": dict(v4_sizes)},
    "stabilizer_strata": {k: {"blocks": stab_counts[k],
                              "mass": sum(mass_of[i] for i in block_of
                                          if label[i] == k)}
                          for k in sorted(stab_counts)},
    "fixed_blocks": [{"item": i,
                      "nnz": sum(1 for r in block_of[i] for x in r if x),
                      "count": mass_of[i]} for i in fixed_blocks],
    "size6_tau_patterns": {" + ".join(k): v for k, v in patterns.items()},
    "size4_fixed_point_types": {" + ".join(k): v for k, v in qpat.items()},
    "negation_duality": {
        "self_negative_orbits": self_neg,
        "paired_orbits": pairs,
        "singleton_pairing": {str(i): fixed_names[i] for i in fixed_blocks},
    },
    "total_mass": TOTAL,
}
with open(OUT, "w") as f:
    json.dump(out, f, indent=1, sort_keys=True)
print("wrote %s" % OUT)
print()
print("ALL CHECKS PASSED" if all(checks) else "SOME CHECKS FAILED")
sys.exit(0 if all(checks) else 1)
