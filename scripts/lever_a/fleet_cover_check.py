#!/usr/bin/env python3
"""Kill-signal coverage check for the canonical-116 census.

Usage: fleet_cover_check.py [--dir PATH_OR_USER@HOST:PATH ...]

Reports, for stonkbot's assignment (the front-to-back pod), how many of its
still-undone items are already covered by shards from other dirs (e.g. a
third machine's run dir, local or remote). When coverage is complete, the
safe-kill condition is met: every queued or in-flight item on stonkbot has a
validating shard elsewhere, so killing it loses only redundant partial work.

Remote dirs are given as user@host:path (listed over ssh, read-only).
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSIGN = os.path.join(ROOT, "scripts", "lever_a", "fleet_assignment_canonical116.json")
LOCAL_DIR = os.path.join(ROOT, "results", "cls_g_csp_shards_L_3blocks_canonical")
STONK_DIR = "~/adinkra-codespace-itemspec/results/cls_g_csp_shards_L_3blocks_canonical"
STONK_SSH = [
    "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
    "-o", "ServerAliveInterval=5", "brandon@192.168.68.71",
]


def shard_items(spec, remote=False):
    """Set of item numbers with shard files under spec (local path or user@host:path)."""
    if remote or (":" in spec and not os.path.isdir(spec)):
        userhost, path = spec.split(":", 1)
        cmd = [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
            "-o", "ServerAliveInterval=5", userhost,
            f"ls {path}/shard_*.json 2>/dev/null | xargs -n1 basename",
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
        except (subprocess.SubprocessError, OSError):
            return set()
        names = out.split()
    else:
        try:
            names = [n for n in os.listdir(spec) if n.startswith("shard_")]
        except OSError:
            return set()
    return {int(n[6:10]) for n in names if n.endswith(".json")}


def main():
    dirs = []
    args = sys.argv[1:]
    if args and args[0] == "--dir":
        dirs = args[1:]
    fa = json.load(open(ASSIGN))
    stonk = sorted(fa["stonkbot"])
    mac = sorted(fa["macm4"])

    stonk_done = shard_items(f"brandon@192.168.68.71:{STONK_DIR}")
    others = shard_items(LOCAL_DIR)  # macm4's dir (plus any rsynced mirrors)
    for d in dirs:
        others |= shard_items(d)

    remaining = [x for x in stonk if x not in stonk_done]
    covered = [x for x in remaining if x in others]
    uncovered = [x for x in remaining if x not in others]
    mac_remaining = [x for x in mac if x not in others]

    print(f"stonkbot: {len(stonk) - len(remaining)}/{len(stonk)} of its items done")
    print(f"stonkbot remaining: {len(remaining)}, covered by others: {len(covered)}, uncovered: {len(uncovered)}")
    print(f"macm4 remaining (local dir): {len(mac_remaining)}")
    if uncovered:
        print(f"first uncovered: {uncovered[:15]}")
        print("not yet safe to kill stonkbot")
    else:
        print("KILL SIGNAL: stonkbot fully covered, safe to kill (only redundant partial work lost)")


if __name__ == "__main__":
    main()
