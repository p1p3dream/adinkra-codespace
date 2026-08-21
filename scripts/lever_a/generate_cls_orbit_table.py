#!/usr/bin/env python3
"""Generate the 116-row CLS 12x12 census orbit table (L side, m=3).

Usage: generate_cls_orbit_table.py [--out PATH]

Reads the Lever A orbit map and every available canonical rep shard (local
canonical dir first, then the stonkbot mirror, then the b300 mirror), validates
each shard structurally (engine/side/blocks/item/complete, class_sum==count,
checksum sum), and writes the orbit table sorted by replicated count. Rows for
reps with no shard yet are rendered as PENDING. At census completion all 116
rows are concrete; this script is the single source for the table, no
hand-typed numbers.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORBIT_MAP = os.path.join(ROOT, "results", "lever_a_slot0_orbits_L_3blocks.json")
DIRS = [
    os.path.join(ROOT, "results", "cls_g_csp_shards_L_3blocks_canonical"),
    os.path.join(ROOT, "results", "cls_g_csp_shards_L_3blocks_canonical_stonkbot_mirror"),
    os.path.join(ROOT, "results", "cls_g_csp_shards_L_3blocks_canonical_b300_mirror"),
]
DEFAULT_OUT = os.path.join(ROOT, "results", "cls-12x12-orbit-table-20260820.md")


def load(item):
    for d in DIRS:
        p = os.path.join(d, f"shard_{item:04d}.json")
        if os.path.exists(p):
            return json.load(open(p)), d
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
    out = DEFAULT_OUT
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
    om = json.load(open(ORBIT_MAP))
    rows, pending, total = [], [], 0
    for o in om["orbits"]:
        rep, size = o["rep_item"], o["size"]
        s, d = load(rep)
        if s is None:
            pending.append(rep)
            rows.append({"rep": rep, "size": size, "members": o["members"], "s": None})
            continue
        s = validate(s, rep, os.path.join(d, f"shard_{rep:04d}.json"))
        total += size * s["count"]
        rows.append({"rep": rep, "size": size, "members": o["members"], "s": s})
    rows.sort(key=lambda r: -(r["size"] * r["s"]["count"]) if r["s"] else -1)

    with open(out, "w") as f:
        f.write("# CLS 12x12 census orbit table (L side, m=3), 2026-08-20\n\n")
        f.write(f"Total over measured orbits: {total:,}"
                + (f" ({len(pending)} orbit(s) pending: {pending})\n\n" if pending else "\n\n"))
        f.write("Sorted by replicated count (orbit_size x count). Checksums are the\n")
        f.write("canonical rep's splitmix64 class-checksum sum.\n\n")
        f.write("| # | rep item | orbit size | count(rep) | replicated | nodes | classes | checksum |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for i, r in enumerate(rows, 1):
            if r["s"] is None:
                f.write(f"| {i} | {r['rep']} | {r['size']} | PENDING | PENDING | - | - | - |\n")
            else:
                s = r["s"]
                f.write(f"| {i} | {r['rep']} | {r['size']} | {s['count']:,} | "
                        f"{r['size']*s['count']:,} | {s['nodes']:,} | {len(s['classes'])} | "
                        f"{s['checksum'][:12]} |\n")
    print(f"wrote {out}: {len(rows)-len(pending)}/116 concrete, pending={pending}, total={total:,}")


if __name__ == "__main__":
    main()
