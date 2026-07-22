#!/usr/bin/env python3
"""Apply reviewed cross-package entity aliases to merged proposals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aliases = json.loads(args.map.read_text(encoding="utf-8"))["aliases"]
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    replacements = 0
    for row in rows:
        for endpoint_name in ("source", "target"):
            endpoint = row[endpoint_name]
            replacement = aliases.get(endpoint["key"])
            if replacement:
                row[endpoint_name] = dict(replacement)
                replacements += 1
    rows.sort(key=lambda row: row["proposal_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"proposals": len(rows), "endpoint_replacements": replacements}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
