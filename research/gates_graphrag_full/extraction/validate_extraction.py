#!/usr/bin/env python3
"""Validate full extraction coverage, anchors, counts, hashes and reports."""

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
    SCHEMA_VERSION,
    atomic_write_text,
    invalid_character_counts,
    load_corpus,
    sha256_file,
    stable_json,
)
from research.gates_graphrag_pilot.extraction.extract_papers import (  # noqa: E402
    count_tokens,
    count_words,
)


SUM_FIELDS = [
    "source_page_count",
    "pages_with_text_count",
    "zero_text_page_count",
    "image_only_page_count",
    "chunk_count",
    "word_count",
    "token_count",
    "anchored_line_count",
    "sectioned_chunk_count",
    "null_replacement_count",
    "replacement_character_count",
    "disallowed_control_character_count",
    "surrogate_character_count",
    "private_use_character_count",
    "pages_without_chunk_count",
]


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: expected an object")
                records.append(value)
    return records


def validate(args: argparse.Namespace) -> int:
    inputs, _ = load_corpus(args.manifest)
    expected = {item.paper.paper_id: item for item in inputs}
    index_rows = load_jsonl(args.output_dir / "extraction_index.jsonl")
    quality = load_json(args.output_dir / "quality_report.json")
    retry = load_json(args.output_dir / "retry_failure_report.json")
    coverage = load_json(args.output_dir / "input_coverage.json")
    determinism = load_json(args.output_dir / "determinism_report.json")

    errors: list[str] = []
    warnings: list[str] = []
    if len(index_rows) != len(expected):
        errors.append(f"index has {len(index_rows)} records, expected {len(expected)}")
    index_ids = [row.get("paper_id") for row in index_rows]
    if len(index_ids) != len(set(index_ids)):
        errors.append("index contains duplicate paper IDs")
    if set(index_ids) != set(expected):
        errors.append("index paper IDs do not exactly match the canonical local-PDF manifest")
    if any(row.get("status") != "success" for row in index_rows):
        errors.append("index contains a non-success record")

    recomputed = {field: 0 for field in SUM_FIELDS}
    seen_chunk_ids: set[str] = set()
    verified_shards = 0
    for index_row in index_rows:
        paper_id = index_row.get("paper_id")
        if paper_id not in expected or index_row.get("status") != "success":
            continue
        item = expected[paper_id]
        shard_value = index_row.get("shard_path")
        if not isinstance(shard_value, str):
            errors.append(f"{paper_id}: shard_path is absent")
            continue
        shard_path = args.output_dir / shard_value
        if not shard_path.is_file():
            errors.append(f"{paper_id}: shard is absent: {shard_value}")
            continue
        if sha256_file(shard_path) != index_row.get("shard_sha256"):
            errors.append(f"{paper_id}: shard SHA-256 does not match the index")
        rows = load_jsonl(shard_path)
        if len(rows) != index_row.get("chunk_count"):
            errors.append(f"{paper_id}: shard chunk count does not match the index")
        actual_source_sha = sha256_file(item.paper.pdf_path)
        if actual_source_sha != index_row.get("source_sha256"):
            errors.append(f"{paper_id}: current source SHA-256 does not match the index")

        expected_chunk_indexes = list(range(len(rows)))
        chunk_indexes = [row.get("chunk_index") for row in rows]
        if chunk_indexes != expected_chunk_indexes:
            errors.append(f"{paper_id}: chunk indexes are not contiguous")
        previous_page = 0
        page_chunk_indexes: dict[int, list[int]] = {}
        texts: list[str] = []
        for row in rows:
            chunk_id = row.get("chunk_id")
            if not isinstance(chunk_id, str) or chunk_id in seen_chunk_ids:
                errors.append(f"{paper_id}: absent or duplicate chunk ID {chunk_id!r}")
            else:
                seen_chunk_ids.add(chunk_id)
            if row.get("schema_version") != SCHEMA_VERSION:
                errors.append(f"{paper_id}: unexpected chunk schema version")
            if row.get("paper_id") != paper_id:
                errors.append(f"{paper_id}: a row has the wrong paper ID")
            page = row.get("page_number")
            if not isinstance(page, int) or page < 1 or page > index_row.get("source_page_count", 0):
                errors.append(f"{paper_id}: invalid page number {page!r}")
                continue
            if page < previous_page:
                errors.append(f"{paper_id}: pages are not ordered")
            previous_page = page
            page_chunk_indexes.setdefault(page, []).append(row.get("page_chunk_index"))
            start = row.get("page_line_start")
            end = row.get("page_line_end")
            if not isinstance(start, int) or not isinstance(end, int) or start < 0 or start > end:
                errors.append(f"{paper_id}: invalid line anchor in {chunk_id}")
            bbox = row.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(x, (int, float)) for x in bbox):
                errors.append(f"{paper_id}: invalid bbox in {chunk_id}")
            section_page = row.get("section_start_page")
            if section_page is not None and (not isinstance(section_page, int) or section_page > page):
                errors.append(f"{paper_id}: invalid section start in {chunk_id}")
            text = row.get("text")
            if not isinstance(text, str) or not text:
                errors.append(f"{paper_id}: absent chunk text in {chunk_id}")
                continue
            if "\x00" in text:
                errors.append(f"{paper_id}: raw U+0000 remains in {chunk_id}")
            if count_words(text) != row.get("word_count"):
                errors.append(f"{paper_id}: word count mismatch in {chunk_id}")
            if count_tokens(text) != row.get("token_count"):
                errors.append(f"{paper_id}: token count mismatch in {chunk_id}")
            provenance = row.get("extraction_provenance", {})
            if provenance.get("source_sha256") != actual_source_sha:
                errors.append(f"{paper_id}: source hash mismatch in {chunk_id}")
            texts.append(text)
        for page, values in page_chunk_indexes.items():
            if values != list(range(len(values))):
                errors.append(f"{paper_id}: page {page} chunk indexes are not contiguous")

        local_counts = {
            "chunk_count": len(rows),
            "word_count": sum(row["word_count"] for row in rows),
            "token_count": sum(row["token_count"] for row in rows),
            "anchored_line_count": sum(row["page_line_end"] - row["page_line_start"] + 1 for row in rows),
            "sectioned_chunk_count": sum(row["section_heading"] is not None for row in rows),
            "null_replacement_count": sum(row["extraction_provenance"]["null_replacement_count"] for row in rows),
            **invalid_character_counts(texts),
        }
        for field in local_counts:
            if local_counts[field] != index_row.get(field):
                errors.append(f"{paper_id}: recomputed {field} does not match the index")
        for field in SUM_FIELDS:
            recomputed[field] += int(index_row.get(field, 0))
        verified_shards += 1

    for field, value in recomputed.items():
        if value != quality.get(field):
            errors.append(f"corpus {field}: index sum {value} != quality report {quality.get(field)}")
    if quality.get("successful_paper_count") != len(expected) or not quality.get("complete"):
        errors.append("quality report does not mark all expected papers complete")
    if retry.get("failure_count") != 0 or retry.get("success_on_first_attempt_count") != len(expected):
        errors.append("retry report does not record clean first-attempt success")
    if not coverage.get("all_downloaded_records_accounted_for"):
        errors.append("input coverage does not account for every downloaded record")
    if coverage.get("downloaded_record_count") != len(expected):
        errors.append("input coverage downloaded count is inconsistent")
    if not determinism.get("all_match") or determinism.get("match_count") != len(expected):
        errors.append("full repeat-extraction determinism check did not match")

    if quality.get("private_use_character_count", 0):
        warnings.append(
            "PDF font mappings emitted private-use characters; counts are preserved for review rather than repaired"
        )
    if quality.get("disallowed_control_character_count", 0):
        warnings.append(
            "PDF text emitted non-NUL control characters; JSON escaped them and the quality report counts them"
        )

    report = {
        "schema_version": BUILD_SCHEMA_VERSION,
        "valid": not errors,
        "expected_paper_count": len(expected),
        "verified_shard_count": verified_shards,
        "verified_chunk_count": len(seen_chunk_ids),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }
    atomic_write_text(args.output_dir / "validation_report.json", stable_json(report) + "\n")
    return 0 if not errors else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.manifest = args.manifest.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    try:
        return validate(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
