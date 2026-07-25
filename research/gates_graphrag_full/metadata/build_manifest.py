#!/usr/bin/env python3
"""Build the deterministic full-corpus Gates literature manifest.

This script reads the curated source collection without modifying it. It
normalizes identifier aliases, verifies canonical PDF artifacts, records exact
duplicate local copies, and emits work-level JSON and flattened CSV manifests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SOURCE_ROOT = Path.home() / "Documents" / "S_James_Gates_Publications"
DEFAULT_PILOT_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "gates_graphrag_pilot"
    / "metadata"
    / "manifest.json"
)
OUTPUT_DIR = Path(__file__).resolve().parent
EXCLUDED_FALSE_AUTHOR_RECORDS = (
    {
        "inspire_id": "2077897",
        "reason": "Incorrectly linked University of Southampton photonics author record.",
    },
    {
        "inspire_id": "2947909",
        "reason": "Incorrectly linked University of Southampton photonics author record.",
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_values(value: str) -> list[str]:
    # The source builder joins values with semicolon plus space. A bare
    # semicolon can be part of an identifier, notably older Wiley DOIs.
    return [part.strip() for part in value.split("; ") if part.strip()]


def ordered_unique(values: Iterable[str], *, casefold: bool = False) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold() if casefold else value
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def pdf_pages(path: Path) -> int:
    result = subprocess.run(
        ["pdfinfo", str(path)], capture_output=True, check=True, text=True
    )
    match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout, re.MULTILINE)
    if not match:
        raise ValueError(f"pdfinfo did not report a page count for {path}")
    return int(match.group(1))


def paper_id(arxiv_ids: list[str], inspire_id: str) -> str:
    return arxiv_ids[0] if arxiv_ids else f"inspire:{inspire_id}"


def source_generated_on(source_root: Path) -> str:
    readme = (source_root / "README.md").read_text()
    match = re.search(r"Generated from INSPIRE author records on (\d{4}-\d{2}-\d{2})", readme)
    if not match:
        raise ValueError("Could not determine the source collection generation date")
    return match.group(1)


def artifact_inventory(source_root: Path, source_records: list[dict[str, Any]]) -> dict[str, Any]:
    canonical_paths = {
        (source_root / "pdfs" / record["pdf_filename"]).resolve()
        for record in source_records
        if record["pdf_status"] == "downloaded"
    }
    canonical_by_hash: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(canonical_paths):
        canonical_by_hash[sha256(path)].append(path)

    duplicate_copies: list[dict[str, Any]] = []
    supporting_documents: list[dict[str, Any]] = []
    for path in sorted(source_root.rglob("*.pdf")):
        resolved = path.resolve()
        if resolved in canonical_paths:
            continue
        digest = sha256(path)
        if digest in canonical_by_hash:
            duplicate_copies.append(
                {
                    "path": str(resolved),
                    "relative_path": str(path.relative_to(source_root)),
                    "sha256": digest,
                    "relationship": "exact_byte_copy",
                    "canonical_paths": [str(p) for p in canonical_by_hash[digest]],
                }
            )
        else:
            supporting_documents.append(
                {
                    "path": str(resolved),
                    "relative_path": str(path.relative_to(source_root)),
                    "sha256": digest,
                    "role": "collection_metadata_not_publication_full_text",
                }
            )

    return {
        "canonical_publication_pdf_count": len(canonical_paths),
        "canonical_publication_pdf_unique_hash_count": len(canonical_by_hash),
        "exact_duplicate_copy_count": len(duplicate_copies),
        "exact_duplicate_copies": duplicate_copies,
        "supporting_document_pdf_count": len(supporting_documents),
        "supporting_document_pdfs": supporting_documents,
    }


def build(source_root: Path, pilot_manifest_path: Path) -> dict[str, Any]:
    source_path = source_root / "MANIFEST.json"
    source_records = json.loads(source_path.read_text())
    pilot = json.loads(pilot_manifest_path.read_text())
    pilot_by_arxiv = {paper["arxiv_id"]: paper for paper in pilot["papers"]}
    inventory = artifact_inventory(source_root, source_records)
    duplicate_artifacts_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for artifact in inventory["exact_duplicate_copies"]:
        duplicate_artifacts_by_hash[artifact["sha256"]].append(artifact)

    papers: list[dict[str, Any]] = []
    duplicate_doi_input_records = 0
    for corpus_order, source in enumerate(source_records, start=1):
        authors = split_values(source["authors"])
        raw_arxiv_ids = split_values(source["arxiv_ids"])
        arxiv_ids = ordered_unique(raw_arxiv_ids)
        raw_dois = split_values(source["dois"])
        dois = ordered_unique(raw_dois, casefold=True)
        report_numbers = ordered_unique(split_values(source["report_numbers"]))
        if len(raw_dois) != len(dois):
            duplicate_doi_input_records += 1

        canonical_id = paper_id(arxiv_ids, source["inspire_id"])
        pilot_record = next(
            (pilot_by_arxiv[item] for item in arxiv_ids if item in pilot_by_arxiv), None
        )

        local_path: Path | None = None
        computed_sha = ""
        size_bytes: int | None = None
        pages: int | None = None
        alternate_artifacts: list[dict[str, Any]] = []
        if source["pdf_status"] == "downloaded":
            local_path = source_root / "pdfs" / source["pdf_filename"]
            computed_sha = sha256(local_path)
            size_bytes = local_path.stat().st_size
            pages = pdf_pages(local_path)
            alternate_artifacts = [
                {
                    "path": item["path"],
                    "relative_path": item["relative_path"],
                    "sha256": item["sha256"],
                    "relationship": item["relationship"],
                }
                for item in duplicate_artifacts_by_hash.get(computed_sha, [])
            ]

        record_flags: list[str] = []
        if len(arxiv_ids) > 1:
            record_flags.append("multiple_arxiv_artifact_aliases")
        if len(raw_dois) != len(dois):
            record_flags.append("duplicate_doi_alias_removed")
        if alternate_artifacts:
            record_flags.append("exact_duplicate_local_artifact")

        papers.append(
            {
                "corpus_order": corpus_order,
                "paper_id": canonical_id,
                "inspire_id": source["inspire_id"],
                "year": int(source["year"]) if source["year"].isdigit() else source["year"],
                "title": source["title"],
                "authors": authors,
                "document_type": source["document_type"],
                "identifiers": {
                    "canonical": canonical_id,
                    "arxiv": arxiv_ids,
                    "doi": dois,
                    "inspire": [source["inspire_id"]],
                    "report_number": report_numbers,
                },
                "identifier_urls": {
                    "arxiv_abstract": [f"https://arxiv.org/abs/{item}" for item in arxiv_ids],
                    "arxiv_pdf": [f"https://arxiv.org/pdf/{item}" for item in arxiv_ids],
                    "doi": [f"https://doi.org/{item}" for item in dois],
                    "inspire": source["inspire_url"],
                },
                "pilot": {
                    "included": pilot_record is not None,
                    "pilot_order": pilot_record["pilot_order"] if pilot_record else None,
                    "series": pilot_record["series"] if pilot_record else None,
                },
                "full_text": {
                    "status": "verified_local_pdf" if local_path else "metadata_only",
                    "canonical_path": str(local_path.resolve()) if local_path else None,
                    "canonical_relative_path": (
                        str(local_path.relative_to(source_root)) if local_path else None
                    ),
                    "filename": source["pdf_filename"] or None,
                    "source": source["pdf_source"] or None,
                    "source_url": source["pdf_source_url"] or None,
                    "bytes": size_bytes,
                    "pages": pages,
                    "sha256": computed_sha or None,
                    "source_manifest_sha256": source["sha256"] or None,
                    "hash_verification": (
                        "matched" if local_path and computed_sha == source["sha256"] else None
                    ),
                    "alternate_local_artifacts": alternate_artifacts,
                    "retrieval_errors": source["retrieval_errors"] or None,
                },
                "artifact_identity": {
                    "work_level_record": True,
                    "arxiv_artifact_aliases": arxiv_ids,
                    "publisher_artifact_aliases": dois,
                    "multiple_arxiv_artifacts": len(arxiv_ids) > 1,
                    "duplicate_record_of": source["duplicate_pdf_of"] or None,
                },
                "record_flags": record_flags,
                "source_record": {
                    "manifest": str(source_path.resolve()),
                    "corpus_order": corpus_order,
                },
            }
        )

    return {
        "schema_version": "1.0",
        "corpus_id": "gates_graphrag_full",
        "generated_on": source_generated_on(source_root),
        "scope": {
            "publication_record_count": len(papers),
            "verified_local_pdf_count": sum(
                paper["full_text"]["status"] == "verified_local_pdf" for paper in papers
            ),
            "metadata_only_record_count": sum(
                paper["full_text"]["status"] == "metadata_only" for paper in papers
            ),
            "pilot_paper_count": sum(paper["pilot"]["included"] for paper in papers),
            "excluded_false_author_record_count": len(EXCLUDED_FALSE_AUTHOR_RECORDS),
            "duplicate_publication_record_count": sum(
                bool(paper["artifact_identity"]["duplicate_record_of"]) for paper in papers
            ),
            "multiple_arxiv_artifact_record_count": sum(
                paper["artifact_identity"]["multiple_arxiv_artifacts"] for paper in papers
            ),
            "source_records_with_repeated_doi_values": duplicate_doi_input_records,
        },
        "curation": {
            "source_manifest": str(source_path.resolve()),
            "source_pdf_directory": str((source_root / "pdfs").resolve()),
            "pilot_manifest": str(pilot_manifest_path.resolve()),
            "paper_id_policy": "First arXiv identifier, including old-style identifiers; otherwise inspire:<INSPIRE ID>.",
            "artifact_policy": "One publication is one work-level record. arXiv identifiers, DOI identifiers, publisher versions, and exact local copies are artifacts or aliases of that record.",
            "identifier_policy": "Preserve source order while removing exact repeated aliases within a record.",
        },
        "excluded_false_author_records": list(EXCLUDED_FALSE_AUTHOR_RECORDS),
        "artifact_inventory": inventory,
        "papers": papers,
    }


CSV_FIELDS = [
    "corpus_order",
    "paper_id",
    "inspire_id",
    "year",
    "title",
    "authors",
    "document_type",
    "arxiv_ids",
    "dois",
    "report_numbers",
    "inspire_url",
    "pilot_included",
    "pilot_order",
    "pilot_series",
    "full_text_status",
    "local_pdf_path",
    "local_pdf_relative_path",
    "local_pdf_filename",
    "local_pdf_bytes",
    "local_pdf_pages",
    "sha256",
    "source_manifest_sha256",
    "hash_verification",
    "pdf_source",
    "pdf_source_url",
    "alternate_local_artifact_count",
    "alternate_local_artifact_paths",
    "multiple_arxiv_artifacts",
    "duplicate_record_of",
    "record_flags",
    "retrieval_errors",
]


def csv_row(paper: dict[str, Any]) -> dict[str, Any]:
    identifiers = paper["identifiers"]
    full_text = paper["full_text"]
    artifacts = full_text["alternate_local_artifacts"]
    return {
        "corpus_order": paper["corpus_order"],
        "paper_id": paper["paper_id"],
        "inspire_id": paper["inspire_id"],
        "year": paper["year"],
        "title": paper["title"],
        "authors": " | ".join(paper["authors"]),
        "document_type": paper["document_type"],
        "arxiv_ids": " | ".join(identifiers["arxiv"]),
        "dois": " | ".join(identifiers["doi"]),
        "report_numbers": " | ".join(identifiers["report_number"]),
        "inspire_url": paper["identifier_urls"]["inspire"],
        "pilot_included": str(paper["pilot"]["included"]).lower(),
        "pilot_order": paper["pilot"]["pilot_order"] or "",
        "pilot_series": paper["pilot"]["series"] or "",
        "full_text_status": full_text["status"],
        "local_pdf_path": full_text["canonical_path"] or "",
        "local_pdf_relative_path": full_text["canonical_relative_path"] or "",
        "local_pdf_filename": full_text["filename"] or "",
        "local_pdf_bytes": full_text["bytes"] or "",
        "local_pdf_pages": full_text["pages"] or "",
        "sha256": full_text["sha256"] or "",
        "source_manifest_sha256": full_text["source_manifest_sha256"] or "",
        "hash_verification": full_text["hash_verification"] or "",
        "pdf_source": full_text["source"] or "",
        "pdf_source_url": full_text["source_url"] or "",
        "alternate_local_artifact_count": len(artifacts),
        "alternate_local_artifact_paths": " | ".join(item["path"] for item in artifacts),
        "multiple_arxiv_artifacts": str(
            paper["artifact_identity"]["multiple_arxiv_artifacts"]
        ).lower(),
        "duplicate_record_of": paper["artifact_identity"]["duplicate_record_of"] or "",
        "record_flags": " | ".join(paper["record_flags"]),
        "retrieval_errors": full_text["retrieval_errors"] or "",
    }


def write_outputs(manifest: dict[str, Any]) -> None:
    json_path = OUTPUT_DIR / "manifest.json"
    csv_path = OUTPUT_DIR / "manifest.csv"
    json_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(csv_row(paper) for paper in manifest["papers"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--pilot-manifest", type=Path, default=DEFAULT_PILOT_MANIFEST)
    args = parser.parse_args()
    write_outputs(build(args.source_root.resolve(), args.pilot_manifest.resolve()))


if __name__ == "__main__":
    main()
