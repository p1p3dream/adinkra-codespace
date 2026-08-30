#!/usr/bin/env python3
"""Assign the undone tail of the canonical-116 census to a joining machine.

Usage: fleet_assign_next.py <pod-name> <threads>

Fleet-lite protocol: items are claimed only by completion (a shard file that
validates), so a new machine simply takes undone items. Stonkbot works its
list front-to-back, so a third machine takes stonkbot's REMAINING items from
the back (front/back meeting in the middle minimizes duplicated in-flight
work; any overlap is wasted compute, never wrong results).

Prints a paste-ready cls-g-csp-shard-items command line and the item spec.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCAL_DIR = os.path.join(ROOT, "results", "cls_g_csp_shards_L_3blocks_canonical")
SB_DIR = "~/adinkra-codespace-itemspec/results/cls_g_csp_shards_L_3blocks_canonical"
SSH = [
    "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
    "-o", "ServerAliveInterval=5", "brandon@192.168.68.71",
]


def done_items(dirpath, remote=False):
    if remote:
        try:
            out = subprocess.run(
                SSH + [f"ls {SB_DIR}/shard_*.json 2>/dev/null | xargs -n1 basename"],
                capture_output=True, text=True, timeout=20,
            ).stdout
        except (subprocess.SubprocessError, OSError):
            return set()
        names = out.split()
    else:
        try:
            names = [n for n in os.listdir(dirpath) if n.startswith("shard_")]
        except OSError:
            return set()
    return {int(n[6:10]) for n in names if n.endswith(".json")}


def spec(v):
    out, s, p = [], v[0], v[0]
    for x in v[1:]:
        if x == p + 1:
            p = x
        else:
            out.append(f"{s}-{p}" if p > s else str(s))
            s = p = x
    out.append(f"{s}-{p}" if p > s else str(s))
    return ",".join(out)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    pod, threads = sys.argv[1], sys.argv[2]
    fa = json.load(open(os.path.join(ROOT, "scripts", "lever_a", "fleet_assignment_canonical116.json")))
    stonk = fa["stonkbot"]
    done = done_items(LOCAL_DIR) | done_items(None, remote=True)
    undone = [x for x in stonk if x not in done]
    if not undone:
        print("nothing left on stonkbot's list; check macm4's tail instead")
        return
    # Take from the BACK (stonkbot grinds front-to-back).
    take = sorted(undone, reverse=True)
    print(f"# {pod}: take {len(take)} undone stonkbot items (back-to-front)")
    print(f"# run dir must be a fresh cls_g_csp_shards_L_3blocks_canonical with the same engine build")
    print(
        f"ADINKRA_CSP_HEARTBEAT=15 ADINKRA_CSP_POD={pod} "
        f"./target-csp/release/adinkra-codespace cls-g-csp-shard-items L 3 {spec(take)} {threads} "
        f"<absolute-path-to>/results/cls_g_csp_shards_L_3blocks_canonical"
    )


if __name__ == "__main__":
    main()
