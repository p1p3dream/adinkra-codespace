#!/usr/bin/env python3
"""Validate the full-corpus Gates citation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def nul_paths(value, path: str = "$") -> list[str]:
    """Return paths to strings containing PostgreSQL-forbidden U+0000."""
    if isinstance(value, str):
        return [path] if "\x00" in value else []
    if isinstance(value, dict):
        return [
            found
            for key, child in value.items()
            for found in nul_paths(child, f"{path}.{key}")
        ]
    if isinstance(value, list):
        return [
            found
            for index, child in enumerate(value)
            for found in nul_paths(child, f"{path}[{index}]")
        ]
    return []


def validate(manifest_path: Path, artifact_dir: Path) -> dict:
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(manifest_payload, dict):
        manifest = manifest_payload.get("papers")
        if not isinstance(manifest, list):
            raise ValueError("canonical source manifest must contain a papers list")
    elif isinstance(manifest_payload, list):
        manifest = manifest_payload
    else:
        raise ValueError("source manifest must contain a papers list or a legacy JSON list")
    corpus_ids = {str(row["inspire_id"]) for row in manifest}
    corpus_paper_ids = {
        str(row.get("paper_id") or f"inspire:{row['inspire_id']}")
        for row in manifest
    }
    expected_local = {
        str(row["inspire_id"])
        for row in manifest
        if (
            isinstance(row.get("full_text"), dict)
            and row["full_text"].get("status") == "verified_local_pdf"
        ) or row.get("pdf_status") == "downloaded"
    }
    expected_local_paper_ids = {
        str(row.get("paper_id") or f"inspire:{row['inspire_id']}")
        for row in manifest
        if str(row["inspire_id"]) in expected_local
    }
    citations = read_jsonl(artifact_dir / "citations.jsonl")
    unresolved = read_jsonl(artifact_dir / "unresolved.jsonl")
    metrics = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
    aliases = json.loads((artifact_dir / "aliases.json").read_text(encoding="utf-8"))
    errors: list[str] = []

    nul_locations: list[str] = []
    for index, row in enumerate(citations, 1):
        nul_locations.extend(f"citations.jsonl:{index}{path[1:]}" for path in nul_paths(row))
    for index, row in enumerate(unresolved, 1):
        nul_locations.extend(f"unresolved.jsonl:{index}{path[1:]}" for path in nul_paths(row))
    for artifact_name, value in (("metrics.json", metrics), ("aliases.json", aliases)):
        nul_locations.extend(f"{artifact_name}{path[1:]}" for path in nul_paths(value))
    for artifact_name in ("citations.jsonl", "unresolved.jsonl", "metrics.json", "aliases.json", "REVIEW.md", "README.md"):
        artifact_path = artifact_dir / artifact_name
        if artifact_path.exists() and b"\x00" in artifact_path.read_bytes():
            nul_locations.append(f"{artifact_name}:raw-byte")
    if nul_locations:
        errors.append(f"PostgreSQL-forbidden U+0000 found at {nul_locations[:20]}")

    if len(manifest) != 295:
        errors.append(f"manifest has {len(manifest)} records, expected 295")
    if len(expected_local) != 166:
        errors.append(f"manifest has {len(expected_local)} local PDFs, expected 166")
    if metrics.get("local_pdfs_processed") != 166:
        errors.append(f"processed {metrics.get('local_pdfs_processed')} PDFs, expected 166")
    if metrics.get("local_pdf_errors") != 0:
        errors.append(f"PDF extraction errors: {metrics.get('local_pdf_errors')}")
    if metrics.get("source_manifest") != str(manifest_path.resolve()):
        errors.append("metrics source_manifest does not match the validation manifest")
    expected_manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if metrics.get("source_manifest_sha256") != expected_manifest_hash:
        errors.append("metrics source_manifest_sha256 does not match the validation manifest")
    expected_contract = "canonical_papers_object" if isinstance(manifest_payload, dict) else "legacy_records_list"
    if metrics.get("source_manifest_contract") != expected_contract:
        errors.append("metrics source_manifest_contract does not match the validation manifest")
    processed_ids = {row["inspire_id"] for row in metrics.get("paper_metrics", []) if row.get("status") == "processed"}
    if processed_ids != expected_local:
        errors.append("processed PDF identifiers do not match the manifest")

    required_citation = {
        "citation_id", "source_paper_id", "target_paper_id", "source_inspire_id",
        "target_inspire_id", "reference_label", "physical_page", "excerpt",
        "resolution_method", "matched_value", "confidence", "review_status",
    }
    citation_ids: set[str] = set()
    methods = Counter()
    for index, row in enumerate(citations, 1):
        missing = required_citation - row.keys()
        if missing:
            errors.append(f"citation row {index} missing {sorted(missing)}")
        if row.get("citation_id") in citation_ids:
            errors.append(f"duplicate citation_id {row.get('citation_id')}")
        citation_ids.add(row.get("citation_id"))
        if row.get("source_inspire_id") not in expected_local:
            errors.append(f"citation row {index} has nonlocal source")
        if row.get("source_paper_id") not in expected_local_paper_ids:
            errors.append(f"citation row {index} has source_paper_id outside local canonical records")
        if row.get("target_inspire_id") not in corpus_ids:
            errors.append(f"citation row {index} target is outside the 295-record corpus")
        if row.get("target_paper_id") not in corpus_paper_ids:
            errors.append(f"citation row {index} target_paper_id is outside canonical records")
        if row.get("source_paper_id") == row.get("target_paper_id"):
            errors.append(f"citation row {index} is a self-edge")
        if not row.get("excerpt") or not row.get("reference_label") or not isinstance(row.get("physical_page"), int):
            errors.append(f"citation row {index} lacks page-anchored evidence")
        method = row.get("resolution_method")
        methods[method] += 1
        expected_status = "pending_title_review" if method == "normalized_title_containment" else "accepted_exact_identifier"
        if row.get("review_status") != expected_status:
            errors.append(f"citation row {index} has inconsistent review status")

    required_stub = {
        "stub_id", "source_paper_id", "source_inspire_id", "reference_label",
        "physical_page", "excerpt", "identifiers", "bibliographic_signals", "review_status",
    }
    stub_ids: set[str] = set()
    for index, row in enumerate(unresolved, 1):
        missing = required_stub - row.keys()
        if missing:
            errors.append(f"unresolved row {index} missing {sorted(missing)}")
        if row.get("stub_id") in stub_ids:
            errors.append(f"duplicate stub_id {row.get('stub_id')}")
        stub_ids.add(row.get("stub_id"))
        signals = row.get("bibliographic_signals") or []
        if "exact_identifier" not in signals and len(signals) < 3:
            errors.append(f"unresolved row {index} has insufficient bibliographic evidence")
        if row.get("review_status") != "unresolved_external":
            errors.append(f"unresolved row {index} has invalid status")
        if row.get("source_paper_id") not in expected_local_paper_ids:
            errors.append(f"unresolved row {index} has source_paper_id outside local canonical records")

    alias_targets = set()
    for mapping_name in ("arxiv_to_inspire", "doi_to_inspire", "normalized_title_to_inspire"):
        for values in aliases.get(mapping_name, {}).values():
            alias_targets.update(values)
    if not alias_targets <= corpus_ids:
        errors.append("aliases contain targets outside the manifest")

    result = {
        "schema_version": "gates-citation-validation-v1",
        "valid": not errors,
        "errors": errors,
        "manifest_records": len(manifest),
        "local_pdfs_processed": metrics.get("local_pdfs_processed"),
        "citation_occurrences": len(citations),
        "distinct_edges": len({(row["source_paper_id"], row["target_paper_id"]) for row in citations}),
        "unresolved_external_stubs": len(unresolved),
        "resolution_methods": dict(sorted(methods.items())),
        "title_matches_pending_review": methods.get("normalized_title_containment", 0),
        "postgres_nul_replacements": metrics.get("postgres_nul_replacements", 0),
        "nul_characters_detected": len(nul_locations),
    }
    (artifact_dir / "VALIDATION.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.manifest.expanduser().resolve(), args.artifact_dir.expanduser().resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
