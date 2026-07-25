#!/usr/bin/env python3
"""Re-extract every source and byte-compare it with its committed shard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.gates_graphrag_full.extraction.build_full_extraction import (  # noqa: E402
    BUILD_SCHEMA_VERSION,
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_DIR,
    atomic_write_text,
    extract_once,
    load_corpus,
    render_rows,
    stable_json,
)


def load_index(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            paper_id = record["paper_id"]
            if paper_id in records:
                raise ValueError(f"duplicate paper_id at {path}:{line_number}: {paper_id}")
            records[paper_id] = record
    return records


def verify(args: argparse.Namespace) -> int:
    inputs, _ = load_corpus(args.manifest)
    index = load_index(args.output_dir / "extraction_index.jsonl")
    results: list[dict[str, Any]] = []
    for position, item in enumerate(inputs, 1):
        record = index.get(item.paper.paper_id)
        if not record or record.get("status") != "success" or not record.get("shard_path"):
            results.append(
                {
                    "paper_id": item.paper.paper_id,
                    "status": "not_verifiable",
                    "reason": "successful extraction index record is absent",
                }
            )
            continue
        shard_path = args.output_dir / record["shard_path"]
        try:
            rows, _ = extract_once(item, args.target_words)
            expected = shard_path.read_bytes()
            actual = render_rows(rows).encode("utf-8")
            same = expected == actual
            results.append(
                {
                    "paper_id": item.paper.paper_id,
                    "status": "match" if same else "mismatch",
                    "shard_path": record["shard_path"],
                    "expected_byte_count": len(expected),
                    "reextracted_byte_count": len(actual),
                }
            )
        except Exception as error:
            results.append(
                {
                    "paper_id": item.paper.paper_id,
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        print(
            f"[{position}/{len(inputs)}] {item.paper.paper_id}: {results[-1]['status']}",
            file=sys.stderr,
        )

    counts = {status: sum(row["status"] == status for row in results) for status in ["match", "mismatch", "error", "not_verifiable"]}
    report = {
        "schema_version": BUILD_SCHEMA_VERSION,
        "method": "repeat full extraction and byte-compare canonical JSONL",
        "target_words": args.target_words,
        "paper_count": len(inputs),
        "all_match": counts["match"] == len(inputs),
        **{f"{key}_count": value for key, value in counts.items()},
        "results": results,
    }
    atomic_write_text(args.output_dir / "determinism_report.json", stable_json(report) + "\n")
    return 0 if report["all_match"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-words", type=int, default=300)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.manifest = args.manifest.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    try:
        return verify(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
