#!/usr/bin/env python3
"""Add evidence-backed, within-pilot arXiv citations to extracted JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_pilot_ids(manifest_path: Path) -> set[str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    papers = payload.get("papers", payload) if isinstance(payload, dict) else payload
    return {str(p["arxiv_id"]).strip() for p in papers if p.get("arxiv_id")}


def enrich(manifest_path: Path, input_path: Path, output_path: Path) -> dict[str, int]:
    pilot_ids = load_pilot_ids(manifest_path)
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    edge_pairs: set[tuple[str, str]] = set()

    for row in rows:
        source = str(row.get("arxiv_id") or row.get("paper_id") or "").removeprefix("arxiv:")
        citations = []
        for target in sorted(pilot_ids - {source}):
            if target not in str(row.get("text") or ""):
                continue
            citations.append(
                {
                    "arxiv_id": target,
                    "locator": f"physical PDF page {row.get('page_number')}",
                    "excerpt": next(
                        (line.strip() for line in str(row.get("text") or "").splitlines() if target in line),
                        target,
                    ),
                    "extraction_method": "exact_arxiv_identifier_match",
                    "confidence": 1.0,
                }
            )
            edge_pairs.add((source, target))
        if citations:
            row["citations"] = citations

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {"chunks": len(rows), "citation_edges": len(edge_pairs)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(enrich(args.manifest, args.input, args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
