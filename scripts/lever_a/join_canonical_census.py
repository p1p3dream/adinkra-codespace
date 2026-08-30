#!/usr/bin/env python3
"""Orbit-aware join of the canonical-116 L m=3 shard census (Lever A).

Usage: join_canonical_census.py <shard_dir> [<shard_dir> ...] [--out PATH]

Each shard dir is a cls-g-csp-shard-items run directory (items.json manifest,
shard_NNNN.json files). Canonical rep shards may be spread across dirs
(local pod + rsynced remote pods); the join collects one shard per canonical
rep, validates it structurally, replicates counts/nodes to orbit-mates via the
proven S V S^-1 bijection (see results/lever-a-symmetry-orbit-analysis-20260818.md),
cross-checks every already-measured anchor against the stride-100 ground truth,
and writes the census artifact.

Semantics of the artifact: totals and class counts are EXACT for the full 825
items (bijection-proven replication); per-class checksums cover canonical
members only (checksums are not equivariant under the group action), so the
artifact records "116 canonical shards + orbit map" rather than per-shard
checksums over all 825 items.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORBIT_MAP = os.path.join(ROOT, "results", "lever_a_slot0_orbits_L_3blocks.json")
ANCHOR_DIR = os.path.join(ROOT, "results", "cls_g_csp_shards_L_3blocks_s100")
DEFAULT_OUT = os.path.join(
    ROOT, "results", "four_color_cls_gmatrix_csp_L_3blocks_orbit_census.json"
)


def load_shard(dirs, item):
    """Return (shard, dir) for item from the first dir that has it; None if absent."""
    for d in dirs:
        p = os.path.join(d, f"shard_{item:04d}.json")
        if os.path.exists(p):
            try:
                s = json.load(open(p))
            except (OSError, json.JSONDecodeError) as e:
                sys.exit(f"REFUSING: unreadable shard {p}: {e}")
            return s, d
    return None, None


def validate(s, item, path):
    assert s["engine"] == "gmatrix_csp.v2-ac3", (path, s.get("engine"))
    assert s["side"] == "L" and s["blocks"] == 3, path
    assert s["item"] == item and s["complete"] is True, path
    csum = 0
    for c in s["classes"]:
        csum = (csum + int(c["checksum"], 16)) & 0xFFFFFFFFFFFFFFFF
    assert sum(c["count"] for c in s["classes"]) == s["count"], path
    assert f"{csum:016x}" == s["checksum"], path
    return s


def main():
    args = [a for a in sys.argv[1:] if a != "--out"]
    out = DEFAULT_OUT
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
    dirs = args
    if not dirs:
        sys.exit(__doc__)

    om = json.load(open(ORBIT_MAP))
    orbits, reps = om["orbits"], om["items_to_run"]
    assert len(orbits) == 116 and len(reps) == 116

    missing, table, conflicts = [], [], []
    total_count = total_nodes = canon_nodes = 0
    hist = {}
    for o in orbits:
        rep = o["rep_item"]
        s, d = load_shard(dirs, rep)
        if s is None:
            missing.append(rep)
            continue
        validate(s, rep, os.path.join(d, f"shard_{rep:04d}.json"))
        # Duplicate independent measurement of the same rep must agree on count.
        for d2 in dirs[dirs.index(d) + 1:]:
            p2 = os.path.join(d2, f"shard_{rep:04d}.json")
            if os.path.exists(p2):
                s2 = validate(json.load(open(p2)), rep, p2)
                if s2["count"] != s["count"] or s2["nodes"] != s["nodes"]:
                    conflicts.append((rep, d, d2, s["count"], s2["count"]))
        size = o["size"]
        total_count += size * s["count"]
        total_nodes += size * s["nodes"]
        canon_nodes += s["nodes"]
        for c in s["classes"]:
            key = (c["nnz"], c["support"], tuple(c["ranks"]))
            hist[key] = hist.get(key, 0) + size * c["count"]
        table.append(
            {
                "orbit_id": o["orbit_id"],
                "rep_item": rep,
                "size": size,
                "members": o["members"],
                "count": s["count"],
                "replicated_count": size * s["count"],
                "nodes": s["nodes"],
            }
        )

    if missing:
        print(f"REFUSING: {len(missing)} canonical rep shards missing: {sorted(missing)[:20]}")
        sys.exit(2)
    if conflicts:
        print(f"REFUSING: count/node conflicts across dirs: {conflicts}")
        sys.exit(2)

    # Belt-and-braces: replicated counts must match every measured anchor
    # (stride-100 ground truth), rep or not.
    checks = []
    for o in orbits:
        rep_s, _ = load_shard(dirs, o["rep_item"])
        for mem in o.get("already_measured", []):
            ap = os.path.join(ANCHOR_DIR, f"shard_{mem:04d}.json")
            if not os.path.exists(ap) or mem == o["rep_item"]:
                continue
            a = json.load(open(ap))
            ok = a["count"] == rep_s["count"] and a["nodes"] == rep_s["nodes"]
            checks.append(
                {"orbit": o["orbit_id"], "rep": o["rep_item"], "anchor": mem, "match": ok}
            )
    bad = [c for c in checks if not c["match"]]
    verdict = "PASS" if all(c["match"] for c in checks) else "FAIL"

    art = {
        "source": "cls L-side m=3 G-matrix census, canonical-116 orbit enumeration",
        "engine": "gmatrix_csp.v2-ac3",
        "semantics": (
            "exact totals via orbit replication (S V S^-1 bijection, proven); "
            "class counts replicated; checksums canonical-only"
        ),
        "side": "L",
        "blocks": 3,
        "count": total_count,
        "replicated_node_total": total_nodes,
        "canonical_node_total": canon_nodes,
        "shards_measured": 116,
        "items_covered": 825,
        "class_histogram": [
            {"nnz": k[0], "support": k[1], "ranks": list(k[2]), "count": v}
            for k, v in sorted(hist.items(), key=lambda kv: -kv[1])
        ],
        "n_classes": len(hist),
        "orbits": table,
        "anchor_cross_checks": checks,
        "anchor_cross_check_verdict": verdict,
        "r_side": (
            "P A_L P^-1 = A_R verified entrywise; R census = image of L under "
            "G -> P G P^-1: counts and class histogram identical, no R search needed"
        ),
        "orbit_map": "results/lever_a_slot0_orbits_L_3blocks.json",
    }
    with open(out, "w") as f:
        json.dump(art, f, indent=1)
    print(f"canonical shards: 116/116 present")
    print(f"L m=3 total count: {total_count:,}")
    print(f"classes: {len(hist)}")
    print(f"anchor cross-checks: {len(checks)}, verdict {verdict}")
    if bad:
        for c in bad:
            print(f"  MISMATCH orbit {c['orbit']} rep {c['rep']} anchor {c['anchor']}")
    print(f"artifact: {out}")


if __name__ == "__main__":
    main()
