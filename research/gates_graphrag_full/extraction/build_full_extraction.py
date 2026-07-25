#!/usr/bin/env python3
"""Build deterministic, per-paper evidence shards for the full Gates corpus.

This is an orchestration layer around the pilot extractor.  It does not alter
the pilot evidence policy, source PDFs, or any database.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import fitz


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.gates_graphrag_pilot.extraction.extract_papers import (  # noqa: E402
    EXTRACTION_STRATEGY,
    SCHEMA_VERSION,
    SECTION_STRATEGY,
    TOKEN_STRATEGY,
    ExtractionError,
    PaperInput,
    extract_paper,
    page_lines,
    sha256_file,
)


BUILD_SCHEMA_VERSION = "gates-full-extraction-build-v1"
DEFAULT_MANIFEST = REPO_ROOT / "research" / "gates_graphrag_full" / "metadata" / "manifest.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CorpusInput:
    paper: PaperInput
    expected_sha256: str
    manifest_row_number: int
    pdf_status: str


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    os.replace(temporary, path)


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def first_arxiv_id(value: str) -> str | None:
    """Preserve the manifest's first arXiv identifier, including old-style IDs."""
    values = [part.strip() for part in value.split(",") if part.strip()]
    return values[0] if values else None


def paper_id_for(row: dict[str, str]) -> str:
    arxiv_id = first_arxiv_id(row.get("arxiv_ids", ""))
    if arxiv_id:
        return arxiv_id
    inspire_id = row.get("inspire_id", "").strip()
    if not inspire_id:
        raise ExtractionError("downloaded manifest row lacks both arXiv and INSPIRE identifiers")
    return f"inspire:{inspire_id}"


def shard_stem(paper_id: str) -> str:
    stem = paper_id.replace("/", "__").replace(":", "__")
    if not stem or stem in {".", ".."}:
        raise ExtractionError(f"unsafe paper ID for shard: {paper_id!r}")
    return stem


def _load_source_records(manifest: Path) -> tuple[list[dict[str, Any]], str]:
    if manifest.suffix.lower() == ".csv":
        with manifest.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle)), "source_csv_v1"
    if manifest.suffix.lower() == ".json":
        with manifest.open(encoding="utf-8") as handle:
            value = json.load(handle)
        records = value.get("papers") if isinstance(value, dict) else None
        if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
            raise ExtractionError(f"{manifest}: expected an object with a papers array")
        return records, "canonical_json_v1"
    raise ExtractionError(f"unsupported manifest format: {manifest.suffix}")


def load_corpus(manifest: Path) -> tuple[list[CorpusInput], dict[str, Any]]:
    manifest = manifest.expanduser().resolve()
    rows, manifest_format = _load_source_records(manifest)

    downloaded: list[CorpusInput] = []
    unavailable: list[dict[str, Any]] = []
    pdf_dir = manifest.parent / "pdfs"
    for row_number, row in enumerate(rows, 2):
        if manifest_format == "canonical_json_v1":
            full_text = row.get("full_text", {})
            status = str(full_text.get("status") or "")
            filename = str(full_text.get("filename") or "")
            raw_path = full_text.get("canonical_path")
            is_downloaded = status == "verified_local_pdf"
            paper_id = str(row.get("paper_id") or "").strip()
            identifier_block = row.get("identifiers", {})
            arxiv_values = identifier_block.get("arxiv", [])
            arxiv_id = str(arxiv_values[0]).strip() if arxiv_values else None
            inspire_id = str(row.get("inspire_id") or "").strip() or None
            title = str(row.get("title") or "").strip() or None
            expected_sha256 = str(full_text.get("sha256") or "").strip()
            pdf_path = Path(str(raw_path)).expanduser().resolve() if raw_path else None
        else:
            status = str(row.get("pdf_status", "")).strip()
            filename = str(row.get("pdf_filename", "")).strip()
            is_downloaded = status == "downloaded"
            paper_id = paper_id_for(row)
            arxiv_id = first_arxiv_id(str(row.get("arxiv_ids", "")))
            inspire_id = str(row.get("inspire_id", "")).strip() or None
            title = str(row.get("title", "")).strip() or None
            expected_sha256 = str(row.get("sha256", "")).strip()
            pdf_path = (pdf_dir / filename).resolve() if filename else None
        if not is_downloaded:
            unavailable.append(
                {
                    "paper_id": str(row.get("paper_id") or "").strip() or None,
                    "inspire_id": inspire_id,
                    "manifest_row_number": row_number,
                    "pdf_status": status or None,
                    "reason": "no downloaded full text in the source manifest",
                }
            )
            continue
        if not filename:
            raise ExtractionError(f"manifest row {row_number}: downloaded row lacks pdf_filename")
        if not paper_id:
            raise ExtractionError(f"manifest row {row_number}: downloaded row lacks canonical paper_id")
        if pdf_path is None:
            raise ExtractionError(f"manifest row {row_number}: downloaded row lacks canonical PDF path")
        downloaded.append(
            CorpusInput(
                paper=PaperInput(
                    paper_id=paper_id,
                    pdf_path=pdf_path,
                    arxiv_id=arxiv_id,
                    inspire_id=inspire_id,
                    title=title,
                ),
                expected_sha256=expected_sha256,
                manifest_row_number=row_number,
                pdf_status=status,
            )
        )

    downloaded.sort(key=lambda item: item.paper.paper_id)
    duplicate_ids = sorted(
        paper_id for paper_id, count in Counter(item.paper.paper_id for item in downloaded).items() if count > 1
    )
    stems = [shard_stem(item.paper.paper_id) for item in downloaded]
    duplicate_stems = sorted(stem for stem, count in Counter(stems).items() if count > 1)
    if duplicate_ids:
        raise ExtractionError(f"duplicate paper IDs: {', '.join(duplicate_ids)}")
    if duplicate_stems:
        raise ExtractionError(f"colliding shard stems: {', '.join(duplicate_stems)}")

    coverage = {
        "schema_version": BUILD_SCHEMA_VERSION,
        "source_manifest": str(manifest),
        "source_manifest_format": manifest_format,
        "source_manifest_sha256": sha256_file(manifest),
        "manifest_record_count": len(rows),
        "downloaded_record_count": len(downloaded),
        "metadata_only_record_count": len(unavailable),
        "metadata_only_records": unavailable,
    }
    return downloaded, coverage


def invalid_character_counts(texts: Iterable[str]) -> dict[str, int]:
    counts = Counter()
    for text in texts:
        for character in text:
            if character == "\ufffd":
                counts["replacement_character"] += 1
            category = unicodedata.category(character)
            if category == "Cc" and character not in "\n\r\t":
                counts["disallowed_control_character"] += 1
            elif category == "Cs":
                counts["surrogate_character"] += 1
            elif category == "Co":
                counts["private_use_character"] += 1
    return {
        "replacement_character_count": counts["replacement_character"],
        "disallowed_control_character_count": counts["disallowed_control_character"],
        "surrogate_character_count": counts["surrogate_character"],
        "private_use_character_count": counts["private_use_character"],
    }


def inspect_pages(pdf_path: Path) -> dict[str, Any]:
    document = fitz.open(pdf_path)
    if document.needs_pass:
        document.close()
        raise ExtractionError(f"encrypted PDF requires a password: {pdf_path}")
    page_records: list[dict[str, Any]] = []
    try:
        for page_index, page in enumerate(document):
            lines = page_lines(page)
            text = "\n".join(line.text for line in lines)
            image_count = len(page.get_images(full=True))
            page_records.append(
                {
                    "page_number": page_index + 1,
                    "page_label": page.get_label() or None,
                    "line_count": len(lines),
                    "text_character_count": len(text),
                    "image_count": image_count,
                    "zero_text": not bool(text.strip()),
                    "image_only": not bool(text.strip()) and image_count > 0,
                }
            )
    finally:
        document.close()
    return {
        "source_page_count": len(page_records),
        "pages_with_text_count": sum(not page["zero_text"] for page in page_records),
        "zero_text_page_count": sum(page["zero_text"] for page in page_records),
        "zero_text_pages": [page["page_number"] for page in page_records if page["zero_text"]],
        "image_only_page_count": sum(page["image_only"] for page in page_records),
        "image_only_pages": [page["page_number"] for page in page_records if page["image_only"]],
        "page_records": page_records,
    }


def render_rows(rows: Sequence[dict[str, Any]]) -> str:
    return "".join(stable_json(row) + "\n" for row in rows)


def extract_once(item: CorpusInput, target_words: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actual_sha256 = sha256_file(item.paper.pdf_path)
    if item.expected_sha256 and actual_sha256 != item.expected_sha256:
        raise ExtractionError(
            f"source checksum mismatch for {item.paper.paper_id}: "
            f"manifest={item.expected_sha256} actual={actual_sha256}"
        )
    rows = list(extract_paper(item.paper, target_words=target_words))
    pages = inspect_pages(item.paper.pdf_path)
    chunk_pages = {int(row["page_number"]) for row in rows}
    text_pages = {
        int(page["page_number"]) for page in pages["page_records"] if not page["zero_text"]
    }
    missing_text_pages = sorted(text_pages - chunk_pages)
    if missing_text_pages:
        raise ExtractionError(
            f"text-bearing pages lack chunks for {item.paper.paper_id}: {missing_text_pages}"
        )
    null_replacements = sum(
        int(row["extraction_provenance"]["null_replacement_count"]) for row in rows
    )
    invalid = invalid_character_counts(row["text"] for row in rows)
    summary = {
        "paper_id": item.paper.paper_id,
        "arxiv_id": item.paper.arxiv_id,
        "inspire_id": item.paper.inspire_id,
        "title": item.paper.title,
        "manifest_row_number": item.manifest_row_number,
        "source_pdf": item.paper.pdf_path.name,
        "source_path": str(item.paper.pdf_path),
        "source_sha256": actual_sha256,
        "manifest_sha256": item.expected_sha256 or None,
        "source_sha256_matches_manifest": not item.expected_sha256 or actual_sha256 == item.expected_sha256,
        "chunk_count": len(rows),
        "word_count": sum(int(row["word_count"]) for row in rows),
        "token_count": sum(int(row["token_count"]) for row in rows),
        "anchored_line_count": sum(
            int(row["page_line_end"]) - int(row["page_line_start"]) + 1 for row in rows
        ),
        "sectioned_chunk_count": sum(row["section_heading"] is not None for row in rows),
        "null_replacement_count": null_replacements,
        **invalid,
        **{key: value for key, value in pages.items() if key != "page_records"},
        "pages_without_chunk_count": pages["source_page_count"] - len(chunk_pages),
        "pages_without_chunks": sorted(set(range(1, pages["source_page_count"] + 1)) - chunk_pages),
        "text_pages_without_chunks": missing_text_pages,
    }
    return rows, summary


def extraction_attempts(
    item: CorpusInput, target_words: int, max_attempts: int
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        start = time.monotonic()
        try:
            rows, summary = extract_once(item, target_words)
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "success",
                    "elapsed_seconds": round(time.monotonic() - start, 6),
                    "error_type": None,
                    "error": None,
                }
            )
            return rows, summary, attempts
        except Exception as error:  # isolate one source without silently losing it
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "failed",
                    "elapsed_seconds": round(time.monotonic() - start, 6),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
    return None, None, attempts


def aggregate_quality(summaries: Sequence[dict[str, Any]], expected: int, failed: int) -> dict[str, Any]:
    sum_fields = [
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
    totals = {field: sum(int(summary[field]) for summary in summaries) for field in sum_fields}
    return {
        "schema_version": BUILD_SCHEMA_VERSION,
        "extractor_schema_version": SCHEMA_VERSION,
        "extractor_strategy": EXTRACTION_STRATEGY,
        "section_strategy": SECTION_STRATEGY,
        "token_strategy": TOKEN_STRATEGY,
        "pymupdf_version": fitz.VersionBind,
        "expected_paper_count": expected,
        "successful_paper_count": len(summaries),
        "failed_paper_count": failed,
        "complete": len(summaries) == expected and failed == 0,
        **totals,
        "paper_summaries": list(summaries),
    }


def build(args: argparse.Namespace) -> int:
    manifest = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    shards_dir = output_dir / "shards"
    inputs, coverage = load_corpus(manifest)
    if shards_dir.exists() and args.clean:
        shutil.rmtree(shards_dir)
    shards_dir.mkdir(parents=True, exist_ok=True)

    index_records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    recovered_after_retry: list[dict[str, Any]] = []

    for position, item in enumerate(inputs, 1):
        rows, summary, attempts = extraction_attempts(item, args.target_words, args.max_attempts)
        shard_name = f"{shard_stem(item.paper.paper_id)}.jsonl"
        shard_path = shards_dir / shard_name
        if rows is None or summary is None:
            failures.append(
                {
                    "paper_id": item.paper.paper_id,
                    "arxiv_id": item.paper.arxiv_id,
                    "inspire_id": item.paper.inspire_id,
                    "source_pdf": item.paper.pdf_path.name,
                    "source_path": str(item.paper.pdf_path),
                    "manifest_row_number": item.manifest_row_number,
                    "attempts": attempts,
                }
            )
            index_records.append(
                {
                    "paper_id": item.paper.paper_id,
                    "status": "failed",
                    "manifest_row_number": item.manifest_row_number,
                    "source_pdf": item.paper.pdf_path.name,
                    "shard_path": None,
                    "attempt_count": len(attempts),
                }
            )
            print(f"[{position}/{len(inputs)}] FAILED {item.paper.paper_id}", file=sys.stderr)
            continue

        rendered = render_rows(rows)
        atomic_write_text(shard_path, rendered)
        shard_sha256 = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        summary.update(
            {
                "shard_path": str(shard_path.relative_to(output_dir)),
                "shard_sha256": shard_sha256,
                "shard_byte_count": len(rendered.encode("utf-8")),
                "attempt_count": len(attempts),
            }
        )
        summaries.append(summary)
        index_records.append({"status": "success", **summary})
        if len(attempts) > 1:
            recovered_after_retry.append(
                {"paper_id": item.paper.paper_id, "attempts": attempts}
            )
        print(
            f"[{position}/{len(inputs)}] {item.paper.paper_id}: "
            f"{summary['source_page_count']} pages, {summary['chunk_count']} chunks",
            file=sys.stderr,
        )

    indexed_ids = [record["paper_id"] for record in index_records]
    expected_ids = [item.paper.paper_id for item in inputs]
    if indexed_ids != expected_ids:
        raise ExtractionError("extraction index does not account for every input in canonical order")

    quality = aggregate_quality(summaries, len(inputs), len(failures))
    retry_report = {
        "schema_version": BUILD_SCHEMA_VERSION,
        "max_attempts": args.max_attempts,
        "retry_trigger": "exception during checksum, open, page inspection, or extraction",
        "input_paper_count": len(inputs),
        "success_on_first_attempt_count": len(inputs) - len(failures) - len(recovered_after_retry),
        "recovered_after_retry_count": len(recovered_after_retry),
        "failure_count": len(failures),
        "recovered_after_retry": recovered_after_retry,
        "failures": failures,
    }
    coverage.update(
        {
            "accounted_downloaded_record_count": len(index_records),
            "successful_extraction_count": len(summaries),
            "failed_extraction_count": len(failures),
            "all_downloaded_records_accounted_for": len(index_records) == len(inputs),
        }
    )
    atomic_write_text(output_dir / "extraction_index.jsonl", render_rows(index_records))
    atomic_write_text(output_dir / "quality_report.json", stable_json(quality) + "\n")
    atomic_write_text(output_dir / "retry_failure_report.json", stable_json(retry_report) + "\n")
    atomic_write_text(output_dir / "input_coverage.json", stable_json(coverage) + "\n")
    return 0 if not failures else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-words", type=int, default=300)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--clean", action="store_true", help="remove existing shards before rebuilding")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.target_words < 50:
        raise SystemExit("--target-words must be at least 50")
    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be at least 1")
    try:
        return build(args)
    except (ExtractionError, OSError, csv.Error, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
