#!/usr/bin/env python3
"""Validate the full Gates literature manifest and write deterministic results."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SOURCE_ROOT = Path.home() / "Documents" / "S_James_Gates_Publications"
EXPECTED_PILOT_IDS = {
    "1911.00807",
    "2002.08502",
    "2006.03609",
    "2007.07390",
    "2012.13308",
    "2012.14015",
    "2304.09830",
    "2311.06842",
    "2407.09334",
}
EXCLUDED_IDS = {"2077897", "2947909"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate() -> dict[str, Any]:
    manifest = json.loads((HERE / "manifest.json").read_text())
    source = json.loads((SOURCE_ROOT / "MANIFEST.json").read_text())
    with (HERE / "manifest.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    with (SOURCE_ROOT / "MISSING_FULL_TEXT.csv").open(newline="") as handle:
        missing_rows = list(csv.DictReader(handle))

    papers = manifest["papers"]
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, observed: Any, expected: Any) -> None:
        checks.append(
            {
                "check": name,
                "status": "pass" if condition else "fail",
                "observed": observed,
                "expected": expected,
            }
        )

    check("source_record_count", len(source) == 295, len(source), 295)
    check("json_record_count", len(papers) == 295, len(papers), 295)
    check("csv_record_count", len(csv_rows) == 295, len(csv_rows), 295)
    check(
        "source_order_preserved",
        [p["inspire_id"] for p in papers] == [r["inspire_id"] for r in source],
        len(papers),
        "all INSPIRE IDs in source order",
    )
    check(
        "titles_preserved",
        [p["title"] for p in papers] == [r["title"] for r in source],
        len(papers),
        "all source titles",
    )

    paper_ids = [p["paper_id"] for p in papers]
    inspire_ids = [p["inspire_id"] for p in papers]
    arxiv_ids = [item for p in papers for item in p["identifiers"]["arxiv"]]
    doi_ids = [item.casefold() for p in papers for item in p["identifiers"]["doi"]]
    check("unique_paper_ids", len(paper_ids) == len(set(paper_ids)), len(set(paper_ids)), 295)
    check("unique_inspire_ids", len(inspire_ids) == len(set(inspire_ids)), len(set(inspire_ids)), 295)
    check("unique_arxiv_aliases", len(arxiv_ids) == len(set(arxiv_ids)), len(set(arxiv_ids)), len(arxiv_ids))
    check("unique_doi_aliases", len(doi_ids) == len(set(doi_ids)), len(set(doi_ids)), len(doi_ids))

    verified = [p for p in papers if p["full_text"]["status"] == "verified_local_pdf"]
    metadata_only = [p for p in papers if p["full_text"]["status"] == "metadata_only"]
    check("verified_local_pdf_count", len(verified) == 166, len(verified), 166)
    check("metadata_only_record_count", len(metadata_only) == 129, len(metadata_only), 129)
    check("source_missing_list_count", len(missing_rows) == 129, len(missing_rows), 129)

    missing_ids = {row["inspire_id"] for row in missing_rows}
    output_missing_ids = {p["inspire_id"] for p in metadata_only}
    check(
        "metadata_only_matches_source_missing_list",
        missing_ids == output_missing_ids,
        len(missing_ids & output_missing_ids),
        129,
    )

    missing_paths: list[str] = []
    non_pdf_headers: list[str] = []
    hash_mismatches: list[str] = []
    size_mismatches: list[str] = []
    canonical_hashes: list[str] = []
    for paper in verified:
        full_text = paper["full_text"]
        path = Path(full_text["canonical_path"])
        if not path.is_file():
            missing_paths.append(str(path))
            continue
        if path.read_bytes()[:5] != b"%PDF-":
            non_pdf_headers.append(str(path))
        digest = file_sha256(path)
        canonical_hashes.append(digest)
        if digest != full_text["sha256"] or digest != full_text["source_manifest_sha256"]:
            hash_mismatches.append(paper["paper_id"])
        if path.stat().st_size != full_text["bytes"]:
            size_mismatches.append(paper["paper_id"])

    check("all_canonical_pdf_paths_exist", not missing_paths, missing_paths, [])
    check("all_canonical_files_have_pdf_header", not non_pdf_headers, non_pdf_headers, [])
    check("all_pdf_hashes_match", not hash_mismatches, hash_mismatches, [])
    check("all_pdf_sizes_match", not size_mismatches, size_mismatches, [])
    check(
        "canonical_pdf_hashes_unique",
        len(canonical_hashes) == len(set(canonical_hashes)),
        len(set(canonical_hashes)),
        166,
    )

    pilot_ids = {p["paper_id"] for p in papers if p["pilot"]["included"]}
    check("pilot_membership", pilot_ids == EXPECTED_PILOT_IDS, sorted(pilot_ids), sorted(EXPECTED_PILOT_IDS))
    pilot_local_count = sum(
        p["pilot"]["included"] and p["full_text"]["status"] == "verified_local_pdf"
        for p in papers
    )
    check("pilot_full_text_count", pilot_local_count == 9, pilot_local_count, 9)

    excluded_present = sorted(EXCLUDED_IDS & set(inspire_ids))
    excluded_reported = {r["inspire_id"] for r in manifest["excluded_false_author_records"]}
    check("false_author_records_absent", not excluded_present, excluded_present, [])
    check("false_author_records_documented", excluded_reported == EXCLUDED_IDS, sorted(excluded_reported), sorted(EXCLUDED_IDS))

    inventory = manifest["artifact_inventory"]
    duplicate_paths_valid = True
    for artifact in inventory["exact_duplicate_copies"]:
        path = Path(artifact["path"])
        if not path.is_file() or file_sha256(path) != artifact["sha256"]:
            duplicate_paths_valid = False
        if not all(file_sha256(Path(p)) == artifact["sha256"] for p in artifact["canonical_paths"]):
            duplicate_paths_valid = False
    check("exact_duplicate_copy_count", inventory["exact_duplicate_copy_count"] == 4, inventory["exact_duplicate_copy_count"], 4)
    check("exact_duplicate_copies_verified", duplicate_paths_valid, duplicate_paths_valid, True)
    check("supporting_metadata_pdf_count", inventory["supporting_document_pdf_count"] == 2, inventory["supporting_document_pdf_count"], 2)

    csv_ids = [row["paper_id"] for row in csv_rows]
    check("csv_json_order_match", csv_ids == paper_ids, len(csv_ids), "all 295 paper IDs in JSON order")
    csv_hashes = {row["paper_id"]: row["sha256"] for row in csv_rows}
    json_hashes = {p["paper_id"]: p["full_text"]["sha256"] or "" for p in papers}
    check("csv_json_hash_match", csv_hashes == json_hashes, len(csv_hashes), 295)

    statuses = Counter(check["status"] for check in checks)
    return {
        "schema_version": "1.0",
        "corpus_id": manifest["corpus_id"],
        "validated_against_collection_generated_on": manifest["generated_on"],
        "status": "pass" if statuses["fail"] == 0 else "fail",
        "summary": {
            "check_count": len(checks),
            "passed": statuses["pass"],
            "failed": statuses["fail"],
            "publication_records": len(papers),
            "verified_local_pdfs": len(verified),
            "metadata_only_records": len(metadata_only),
            "canonical_pdf_pages": sum(p["full_text"]["pages"] or 0 for p in papers),
        },
        "checks": checks,
    }


def main() -> None:
    result = validate()
    (HERE / "VALIDATION.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
